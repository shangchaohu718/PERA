"""Independent robot histories with serial expert dispatch through one shared Actor."""

from collections import defaultdict
from threading import Lock

import torch


class NonfinitePolicyAction(FloatingPointError):
    def __init__(self, robot_ids):
        self.robot_ids = tuple(robot_ids)
        super().__init__(f"Nonfinite shared-policy actions for robots {self.robot_ids}")


class MultiRobotController:
    """One bank, one history per robot, and one enabled task expert per inference call.

    The caller advances a single shared physics world after receiving the complete
    action batch. Scene selection is explicit; this is not autonomous task discovery.
    """

    def __init__(self, actor, experts):
        if not experts:
            raise ValueError("Provide at least one robot")
        if any(name is not None and name not in actor.expert_names for name in experts):
            raise ValueError("Unknown task expert")
        self.actor = actor
        self.experts = tuple(experts)
        self.history = None
        self.pending_reset = torch.ones(len(experts), dtype=torch.bool)
        self.last_dispatch = []
        self._lock = Lock()

    def reset(self, ids=None):
        if ids is None:
            self.pending_reset.fill_(True)
        else:
            ids = torch.as_tensor(ids, dtype=torch.long)
            if bool(((ids < 0) | (ids >= len(self.experts))).any()):
                raise ValueError("Unknown robot index")
            self.pending_reset[ids] = True

    @torch.no_grad()
    def infer(self, observation, goal, contexts=None):
        n = len(self.experts)
        device = next(self.actor.pi.parameters()).device
        if (
            observation.shape != (n, 608)
            or goal.shape != (n, 730)
            or observation.device != device
            or goal.device != device
        ):
            raise ValueError(
                "Expected native [N,608] observations and [N,730] goals on Actor device"
            )
        if not bool(observation.isfinite().all() and goal.isfinite().all()):
            raise ValueError("Nonfinite policy input")
        contexts = [None] * n if contexts is None else contexts
        if len(contexts) != n:
            raise ValueError("Supply one context per robot")
        groups = defaultdict(list)
        for i, name in enumerate(self.experts):
            dim = self.actor.expert_specs[name]["context_dim"] if name else 0
            value = contexts[i]
            if dim and (
                not isinstance(value, torch.Tensor)
                or value.shape != (dim,)
                or value.device != device
                or not value.is_floating_point()
                or not bool(value.isfinite().all())
            ):
                raise ValueError(f"Robot {i} requires fresh context of width {dim}")
            groups[name].append(i)
        # Another user of this controller may not interleave its expert selection.
        with self._lock:
            history = (
                observation[:, None].expand(-1, 10, -1).clone()
                if self.history is None
                else torch.cat((self.history[:, 1:], observation[:, None]), 1)
            )
            reset = self.pending_reset.to(device)
            history[reset] = observation[reset, None].expand(-1, 10, -1)
            actions = observation.new_empty((n, 29))
            dispatch = []
            try:
                for expert, rows in groups.items():
                    ids = torch.tensor(rows, device=device, dtype=torch.long)
                    self.actor.select_expert(expert)
                    dim = self.actor.context_dim
                    context = torch.stack([contexts[i] for i in rows]) if dim else None
                    actions[ids] = self.actor(history[ids], goal[ids], context)
                    dispatch.append(
                        dict(
                            expert=expert,
                            robots=rows,
                            selected=self.actor.pi.moe.last_selected.flatten().cpu().tolist(),
                            contribution=self.actor.pi.moe.last_weight.flatten().cpu().tolist(),
                        )
                    )
            finally:
                self.actor.select_expert(None)
            if not bool(actions.isfinite().all()):
                raise NonfinitePolicyAction(
                    (~actions.isfinite()).any(-1).nonzero().flatten().tolist()
                )
            self.history = history
            self.pending_reset.zero_()
            self.last_dispatch = dispatch
            return actions
