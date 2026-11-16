"""One physical clock with independent 50Hz and native ULTRA 60Hz control."""

import numpy as np
import mujoco
import torch

from pera.interaction_showcase import Arena


class MultiRateArena(Arena):
    def __init__(self, config, actor):
        super().__init__(config, actor)
        if not np.isclose(config["physics_dt"], 1 / 1020):
            raise ValueError("The scheduler retains the exact native ULTRA 1020Hz physics clock")
        self.microtick = 0
        self.next_due = [0] * len(self.slots)
        self.actual_physics_steps = 0
        self.periods = [85 if s.row["kind"] == "box" else 102 for s in self.slots]
        self.control_counts = [0] * len(self.slots)

    def reset_due(self, i, repeat):
        s = self.slots[i]
        end = s.horizon if s.reference is not None else s.row.get("repeat_steps")
        completed_goal = s.row.get("loop_on_goal", False) and s.goal_ready
        if repeat and (completed_goal or (end is not None and s.episode_step >= end)):
            self.reset_events.append(
                dict(
                    task=s.name,
                    time_s=float(self.data.time),
                    tracking_lost=bool(s.lost),
                    reason="goal_completed_dwell_1s"
                    if completed_goal
                    else "reference_boundary"
                    if s.reference is not None
                    else "timed_skill_loop",
                    goal_metrics=s.records[-1] if completed_goal else None,
                )
            )
            s.reset_native()
            s.episode += 1
            s.map.push_reset(self)
            self.controller.reset([i])
            mujoco.mj_forward(self.model, self.data)
            s.map.pull(self)

    @torch.no_grad()
    def step(self, repeat=True):
        tennis = next(s for s in self.slots if s.row["kind"] == "tennis")
        boxes = [s for s in self.slots if s.row["kind"] == "box"]
        while self.microtick < (self.tick + 1) * 102:
            due = [i for i, target in enumerate(self.next_due) if self.microtick >= target]
            if due:
                for i in due:
                    self.reset_due(i, repeat)
                for i in due:
                    s = self.slots[i]
                    if s.row["kind"] not in ("tennis", "box"):
                        s.map.pull(self)
                inputs = [self.slots[i].inputs(self) for i in due]
                obs = torch.stack([x[0] for x in inputs])
                goal = torch.stack([x[1] for x in inputs])
                actions = self.controller.infer(obs, goal, [x[2] for x in inputs], robot_ids=due)
                for i, action in zip(due, actions):
                    self.slots[i].execute(
                        action, self, raw_action=self.controller.last_raw_actions[i]
                    )
                    self.next_due[i] += self.periods[i]
                self.sync_markers()
            if not tennis.native.ball_released:
                tennis.native._set_ball_state(
                    tennis.native.incoming_ball_plan.launch_position_w,
                    tennis.native._planned_launch_velocity_w,
                )
                tennis.native.ball_released = True
                tennis.map.push_ball(self)
            for s in boxes:
                s.native.torque(self)
                if (self.microtick + 5) % 85 == 0:
                    s.map.pull(self)
            tennis.map.pull(self)
            tennis.native._apply_ball_aerodynamics()
            self.data.xfrc_applied[tennis.map.bmap[tennis.native.ball_body_id]] = (
                tennis.data.xfrc_applied[tennis.native.ball_body_id]
            )
            mujoco.mj_step(self.model, self.data)
            tennis.map.pull(self, forward=False)
            pending = tennis.native._ground_impact_pending
            tennis.native._apply_explicit_ground_rebound()
            self.ball_rebounds += int(pending and not tennis.native._ground_impact_pending)
            tennis.native._ball_racket_contact()
            tennis.map.push_ball(self)
            self.microtick += 5
            self.actual_physics_steps += 1
            finished = [i for i, target in enumerate(self.next_due) if self.microtick >= target]
            for i in finished:
                s = self.slots[i]
                s.map.pull(self, forward=s.row["kind"] not in ("tennis", "box"))
                s.after_control(self)
                self.control_counts[i] += 1
                s.records[-1]["expert_contribution"] = self.controller.last_dispatch[i][
                    "contribution"
                ][0]
            if any(self.slots[i] is tennis for i in finished):
                tennis.audit.check(float(self.data.time))
                sim = tennis.native
                sim.elapsed_s = self.data.time - tennis.cycle_start
                if sim.elapsed_s >= sim.episode_duration_s:
                    tennis.audit.end_cycle(True)
                    tennis.audit.begin()
                    before = self.data.qpos[tennis.map.qmap[:36]].copy()
                    sim.start_next_strike_cycle()
                    self.sync_markers()
                    sim._set_ball_state(sim.incoming_ball_plan.launch_position_w)
                    sim.ball_released = False
                    tennis.map.push_ball(self)
                    tennis.cycle_start = float(self.data.time)
                    self.ball_launches += 1
                    assert np.array_equal(before, self.data.qpos[tennis.map.qmap[:36]])
        self.tick += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite shared physical state")
