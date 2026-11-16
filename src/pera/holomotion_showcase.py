"""Official HoloMotion weights, native domain FFNs, isolated per-robot memory."""

import hashlib
import json
from types import SimpleNamespace

import torch
from torch import nn

from pera.multi_robot import NonfinitePolicyAction
from pera.paths import ROOT
from pera.robot import get_robot


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ActionConvention:
    """Named conversion between the renderer's SDK targets and official actions."""

    def __init__(self, pi, device):
        self.pi = pi
        contract = json.loads((ROOT / "configs/g1_actor_io_contract.json").read_text())
        self.pi_contract_names = contract["joint_names"]
        sdk = get_robot("g1").joint_names
        self.sdk_from_holo = torch.tensor(
            [contract["joint_names"].index(n) for n in sdk], device=device
        )
        self.holo_from_sdk = torch.argsort(self.sdk_from_holo)
        self.old_offset = torch.tensor(contract["action_offset"], device=device)
        self.old_scale = torch.tensor(contract["action_scale"], device=device)
        self.default = torch.tensor(contract["default_joint_pos"], device=device)

    def old_to_sdk(self, action):
        target = (self.old_offset + self.old_scale * action)[..., self.sdk_from_holo]
        return (target - self.pi.ref_q0) / self.pi.ref_alpha

    def sdk_to_old(self, action):
        target = (self.pi.ref_q0 + self.pi.ref_alpha * action)[..., self.holo_from_sdk]
        return (target - self.old_offset) / self.old_scale


class OfficialShowcaseActor(nn.Module):
    official_holomotion = True

    def __init__(self, native_root, device, base_dir=None, bank_path=None):
        super().__init__()
        from modumotion_g1.official import load_official
        from modumotion_g1.domain_moe import load_domain, public_digest, set_active_domains

        self.policy = json.loads((ROOT / "configs/demo_policy.json").read_text())
        if bank_path is not None:
            from pera.holomotion_bank import FrozenHoloMotionBank

            bank = FrozenHoloMotionBank.load(bank_path, device, base_dir)
            self.actor = bank.actor
            self.domains = bank.domains
            self.policy["experts"] = bank.records
            self.expert_specs = {
                name: dict(context_dim=record["context"]["dim"] if record.get("context") else 0)
                for name, record in bank.records.items()
            }
        else:
            base = base_dir or native_root / self.policy["base_relative_to_native_root"]
            for name, record in self.policy["base_files"].items():
                if digest(base / name) != record["sha256"]:
                    raise ValueError(f"Official base hash mismatch: {name}")
            self.actor, critic, _ = load_official(device=device, checkpoint=base)
            del critic
            self.domains = {}
            self.expert_specs = {}
            for name, record in self.policy["experts"].items():
                relative = (
                    record.get("path", record["source_relative_to_native_root"])
                    if native_root.resolve() == ROOT
                    else record["source_relative_to_native_root"]
                )
                filename = native_root / relative
                if digest(filename) != record["sha256"]:
                    raise ValueError(f"Official expert hash mismatch: {name}")
                ctx = record["context"]
                self.domains[name] = load_domain(
                    self.actor,
                    filename,
                    **(
                        dict(
                            context_dim=ctx["dim"],
                            context_schema=ctx["schema"],
                            encoder_sha256=ctx["encoder_sha256"],
                        )
                        if ctx
                        else {}
                    ),
                )
                self.expert_specs[name] = dict(context_dim=ctx["dim"] if ctx else 0)
        self.actor.eval().requires_grad_(False)
        self.public_sha256 = public_digest(self.actor)
        if self.public_sha256 != self.policy["public_sha256"]:
            raise ValueError("Actor is not the selected official HoloMotion public base")
        set_active_domains(self.actor, [])
        robot = get_robot("g1")
        self.pi = SimpleNamespace(
            ref_q0=torch.tensor(robot.default_joint_pos_list(), device=device),
            ref_alpha=torch.tensor(robot.action_scale_list(), device=device),
        )
        self.adapter = ActionConvention(self.pi, device)
        self.expert_names = tuple(self.domains)

    def reference_actor(self):
        return self.adapter

    def select_expert(self, name):
        from modumotion_g1.domain_moe import set_active_domains, set_domain_context

        set_domain_context(self.actor, None)
        set_active_domains(self.actor, [self.domains[name]] if name else [])

    def verify(self):
        from modumotion_g1.domain_moe import public_digest

        if public_digest(self.actor) != self.public_sha256:
            raise ValueError("Frozen official Actor changed")
        for name, module in self.actor.named_parameters():
            if module.requires_grad:
                raise ValueError(f"Demonstration parameter requires gradients: {name}")
        return self.public_sha256


class OfficialMultiRobotController:
    """Serial dispatch through one Actor; each robot owns all temporal state."""

    CACHE_KEYS = (
        "_k_cache",
        "_v_cache",
        "_ref_hist_k_cache",
        "_ref_hist_v_cache",
        "_kv_cache_len",
        "_kv_cache_write_idx",
        "_kv_cache_abs_pos",
        "_prev_last_moe_router_p",
        "_prev_last_moe_router_valid",
    )

    def __init__(self, actor, experts):
        self.actor, self.experts = actor, tuple(experts)
        self.caches = [None] * len(experts)
        self.last_dispatch = [None] * len(experts)
        self.last_raw_actions = [None] * len(experts)
        self.validate_on_next_infer = False
        self.isolation_report = None
        self.device = next(actor.parameters()).device
        self.reset()

    def _snapshot(self):
        module = self.actor.actor.actor_module
        return {k: getattr(module, k) for k in self.CACHE_KEYS if hasattr(module, k)}

    def _restore(self, state):
        module = self.actor.actor.actor_module
        for key, value in state.items():
            setattr(module, key, value)

    def reset(self, ids=None):
        for i in range(len(self.experts)) if ids is None else ids:
            self.actor.actor.reset_kv_cache(1, self.device)
            self.actor.actor.clear_env_cache(None)
            self.caches[i] = self._snapshot()

    @torch.no_grad()
    def infer(self, observation, goal, contexts=None, robot_ids=None):
        from modumotion_g1.domain_moe import set_domain_context

        if self.validate_on_next_infer:
            self.validate_on_next_infer = False
            self.isolation_report = self.check_isolation(observation, goal, contexts)
        actions = []
        robot_ids = list(range(len(self.experts))) if robot_ids is None else list(robot_ids)
        dispatch = list(self.last_dispatch)
        try:
            for j, i in enumerate(robot_ids):
                expert = self.experts[i]
                self._restore(self.caches[i])
                self.actor.select_expert(expert)
                dim = self.actor.expert_specs[expert]["context_dim"] if expert else 0
                context = contexts[j]
                if dim:
                    if context.shape != (dim,) or not bool(context.isfinite().all()):
                        raise ValueError(f"Invalid context for {expert}")
                    set_domain_context(
                        self.actor.actor, {self.actor.domains[expert]: context[None]}
                    )
                result = self.actor.actor(observation[j : j + 1], mode="inference")["actions"]
                if not bool(result.isfinite().all()):
                    raise NonfinitePolicyAction([i])
                self.last_raw_actions[i] = result[0].detach().clone()
                actions.append(self.actor.adapter.old_to_sdk(result)[0])
                self.caches[i] = self._snapshot()
                weight = self.actor.actor.actor_module.layers[0].last_group_weights
                dispatch[i] = dict(
                    expert=expert, robots=[i], contribution=[float(weight[..., -1].mean())]
                )
        finally:
            self.actor.select_expert(None)
        self.last_dispatch = dispatch
        return torch.stack(actions)

    @torch.no_grad()
    def check_isolation(self, observation, goal, contexts, steps=8):
        shared = OfficialMultiRobotController(self.actor, self.experts)
        singles = [OfficialMultiRobotController(self.actor, [e]) for e in self.experts]
        reverse = OfficialMultiRobotController(self.actor, self.experts[::-1])
        ids = torch.arange(len(self.experts) - 1, -1, -1, device=self.device)
        maximum = reverse_maximum = 0.0
        for _ in range(steps):
            joint = shared.infer(observation, goal, contexts)
            reversed_actions = reverse.infer(observation[ids], goal[ids], contexts[::-1]).flip(0)
            single = torch.stack(
                [
                    c.infer(observation[i : i + 1], goal[i : i + 1], [contexts[i]])[0]
                    for i, c in enumerate(singles)
                ]
            )
            maximum = max(maximum, float((joint - single).abs().max()))
            reverse_maximum = max(reverse_maximum, float((joint - reversed_actions).abs().max()))
        addresses = [s["_k_cache"].data_ptr() for s in shared.caches]
        if maximum > 1e-6 or reverse_maximum > 1e-6 or len(set(addresses)) != len(addresses):
            raise RuntimeError("Official multi-robot memory/context isolation check failed")
        return dict(
            steps=steps,
            actual_official_weights=True,
            single_vs_shared_action_max_abs=maximum,
            forward_vs_reverse_action_max_abs=reverse_maximum,
            distinct_attention_cache_storage=True,
        )
