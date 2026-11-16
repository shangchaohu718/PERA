"""One frozen G1 Actor, multiple installed experts, at most one enabled expert."""

from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from pera.context import ContextMoE, ContextPlugin
from pera.paths import ROOT
from pera.routing import (
    CompetitiveMoE,
    Plugin,
    features,
    load_base,
    policy_from_args,
    public_digest,
)

FORMAT = "pera_g1_switched_expert_bank_v1"
CONTEXT_FORMAT = "blink_g1_competitive_suite31_v1"
PLAIN_FORMATS = {
    "blink_g1_eight_plus_one_competitive_v1",
    "blink_g1_martial_seven_ppo_competitive_v1",
    "blink_g1_stairs_routing_ab_v1",
}


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _finite(state):
    return all(
        not value.is_floating_point() or bool(value.isfinite().all())
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _portable(value):
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and value.startswith("/"):
        path = Path(value)
        return (
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else "${EXTERNAL}/" + path.name
        )
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def _plugin(kind, context_dim):
    if kind == "context":
        if type(context_dim) is not int or context_dim < 0:
            raise ValueError("Context dimension must be a nonnegative integer")
        return ContextPlugin(context_dim)
    if kind == "plain" and context_dim == 0:
        return Plugin()
    raise ValueError("Unsupported competitive expert kind")


def compose_config(path, *, device="cpu"):
    """Load a portable composition manifest; paths are relative to the release root."""
    config = json.loads(Path(path).read_text())
    if config.get("max_active_experts") != 1:
        raise ValueError("This bank supports at most one active task expert")

    def resolve(value):
        value = Path(value)
        return value if value.is_absolute() else ROOT / value

    historical = {}
    if config.get("weights_manifest"):
        manifest = json.loads(resolve(config["weights_manifest"]).read_text())
        historical = {row["path"]: row for row in manifest["resources"]}
    experts, metadata = {}, {}
    for row in config["experts"]:
        name = row["name"]
        if name in experts:
            raise ValueError(f"Duplicate expert name: {name}")
        experts[name] = resolve(row["path"])
        if file_digest(experts[name]) != row["sha256"]:
            raise ValueError(f"Expert file fingerprint mismatch: {name}")
        record = historical.get(row["path"], {})
        metadata[name] = {
            key: value
            for key, value in record.items()
            if key.startswith("historical_") or key in ("evaluation_source", "acceptance_source")
        }
    base_path = resolve(config["base"])
    if file_digest(base_path) != config["base_sha256"]:
        raise ValueError("Base file fingerprint mismatch")
    return SwitchedActor.from_checkpoints(base_path, experts, device=device, metadata=metadata)


class SwitchedMoE(ContextMoE):
    """Reuse the original routing calculation for the one explicitly enabled expert."""

    @property
    def plugin(self):
        if self._active_expert is None:
            raise RuntimeError("No task expert is enabled")
        return self.plugins[self._active_expert]

    @property
    def domain_enabled(self):
        return self._active_expert is not None

    @domain_enabled.setter
    def domain_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise TypeError("Expert enabled must be a bool")
        if not enabled:
            self.select_expert(None)
        elif self._active_expert is None:
            raise ValueError("Choose a named expert before enabling it")

    def select_expert(self, name):
        if name is not None and name not in self.plugins:
            raise ValueError(f"Unknown expert: {name}")
        if name != self._active_expert:
            self._active_expert = name
            self._generation += 1
            self.runtime_context = None
            self.last_features = {}
            self.last_selected = self.last_weight = self.last_margin = None

    def compete(self, f):
        if self._active_expert is None:
            zero = f["hidden"].new_zeros((len(f["hidden"]), 1))
            return f["native"], zero.bool(), zero, zero
        spec = self.expert_specs[self._active_expert]
        if spec["kind"] == "plain":
            return CompetitiveMoE.compete(self, f)
        context = f.get("context", self.runtime_context)
        dim = spec["context_dim"]
        if context is None and dim:
            raise ValueError(f"Expert {self._active_expert!r} requires fresh [B,{dim}] context")
        if context is not None and (
            not isinstance(context, torch.Tensor)
            or not context.is_floating_point()
            or context.shape != (len(f["hidden"]), dim)
            or context.device != f["hidden"].device
        ):
            raise ValueError("Active expert context shape, device or dtype mismatch")
        return ContextMoE.compete(self, f)

    def forward(self, x, r, routed=True):
        result = super().forward(x, r, routed)
        self.last_features["bank_generation"] = torch.full(
            (len(x), 1), self._generation, device=x.device, dtype=torch.long
        )
        return result

    def predict(self, cache, head):
        generation = cache.get("bank_generation")
        if generation is None or not bool((generation == self._generation).all()):
            raise ValueError("Cached features predate an expert switch; run a fresh forward")
        return super().predict(cache, head)


class SwitchedActor(nn.Module):
    """An inference Actor containing one public policy and all installed task experts.

    Inputs are native G1 observation history [B,10,608], reference goal [B,730],
    and the active task's original frozen context [B,C]. Outputs are SDK actions.
    Switches are global to this Actor instance, not different experts per environment.
    """

    def __init__(self, pi, args, robot, plugins, specs, base_sha256):
        super().__init__()
        if not plugins or set(plugins) != set(specs):
            raise ValueError("Provide a nonempty matching expert/specification dictionary")
        for name in plugins:
            if not isinstance(name, str) or not name or "." in name:
                raise ValueError("Expert names must be nonempty module keys without dots")
        before = public_digest(pi)
        if "plugin" in pi.moe._modules:
            del pi.moe.plugin
        pi.moe.__class__ = SwitchedMoE
        pi.moe.__dict__.pop("domain_enabled", None)
        pi.moe.plugins = nn.ModuleDict(plugins)
        pi.moe.expert_specs = copy.deepcopy(specs)
        pi.moe._active_expert = None
        pi.moe._generation = 0
        pi.moe.runtime_context = None
        pi.moe.last_features = {}
        self.pi = pi
        self.args = copy.deepcopy(args)
        self.robot = robot
        self.base_sha256 = base_sha256
        self.public_sha256 = before
        self.expert_specs = copy.deepcopy(specs)
        self.eval().requires_grad_(False)
        if public_digest(self.pi) != before:
            raise ValueError("Installing experts changed the public Actor")

    @property
    def expert_names(self):
        return tuple(self.pi.moe.plugins.keys())

    @property
    def active_expert(self):
        return self.pi.moe._active_expert

    @property
    def context_dim(self):
        return self.expert_specs[self.active_expert]["context_dim"] if self.active_expert else 0

    @property
    def switches(self):
        return {name: name == self.active_expert for name in self.expert_names}

    @property
    def switch_generation(self):
        return self.pi.moe._generation

    def select_expert(self, name=None):
        """Atomically turn the previous expert off and select one expert, or none."""
        self.pi.moe.select_expert(name)

    def set_expert_enabled(self, name, enabled):
        """An individual switch; enabling a second expert is rejected without mutation."""
        if name not in self.expert_specs:
            raise ValueError(f"Unknown expert: {name}")
        if not isinstance(enabled, bool):
            raise TypeError("Expert enabled must be a bool")
        if enabled:
            if self.active_expert is not None and self.active_expert != name:
                raise ValueError(
                    "At most one task expert can be enabled; use select_expert to switch"
                )
            self.select_expert(name)
        elif self.active_expert == name:
            self.select_expert(None)

    def set_switches(self, switches):
        """Set a complete selection; omitted names are off. Validate before applying."""
        if not isinstance(switches, Mapping) or not set(switches) <= set(self.expert_names):
            raise ValueError("Switch mapping contains unknown experts")
        if any(not isinstance(value, bool) for value in switches.values()):
            raise TypeError("Switch values must be bools")
        enabled = [name for name, value in switches.items() if value]
        if len(enabled) > 1:
            raise ValueError("At most one task expert can be enabled")
        self.select_expert(enabled[0] if enabled else None)

    @torch.no_grad()
    def forward(self, history, goal, context=None):
        self.pi.moe.runtime_context = context if self.active_expert is not None else None
        try:
            return self.pi(history, goal)
        finally:
            self.pi.moe.runtime_context = None

    def cached_features(self, goal):
        if not self.pi.moe.last_features:
            raise ValueError("Run a fresh forward after selecting an expert")
        return features(self.pi, goal)

    @torch.no_grad()
    def predict_cached(self, cache):
        return self.pi.moe.predict(cache, self.pi.dec[-1])

    def reference_actor(self):
        """Optional original observation/action adapter for existing native task bridges."""
        from experiments.g1_competitive_suite31_v1.model import Actor

        device = next(self.pi.parameters()).device
        return Actor(None, 0, "", device=device, bank=self)

    def save_bundle(self, path):
        """Save a self-contained inference Actor; runtime switches load all-off."""
        if public_digest(self.pi) != self.public_sha256:
            raise ValueError("Public Actor changed since composition")
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        state = {key: value.detach().cpu() for key, value in self.pi.state_dict().items()}
        if not _finite(state):
            raise ValueError("Cannot save nonfinite model parameters")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        if temporary.exists():
            raise FileExistsError(temporary)
        payload = dict(
            format=FORMAT,
            robot="g1",
            args=_portable(self.args),
            pi=state,
            experts=_portable(self.expert_specs),
            base_sha256=self.base_sha256,
            public_sha256=self.public_sha256,
            max_active_experts=1,
            default_active_expert=None,
            context_encoders="External frozen task resources, not synthesized or included here",
        )
        try:
            torch.save(payload, temporary)
            if path.exists():
                raise FileExistsError(path)
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def from_checkpoints(cls, base_path, experts, *, device="cpu", metadata=None):
        base_sha = file_digest(base_path)
        pi, args, robot = load_base(base_path, device=device)
        digest = public_digest(pi)
        plugins, specs = {}, {}
        for name, path in experts.items():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("public_sha256") != digest:
                raise ValueError(f"Expert {name!r} has another public Actor")
            for key in ["base_sha256", "base_checkpoint_sha256", "public_checkpoint_sha256"]:
                if key in payload and payload[key] != base_sha:
                    raise ValueError(f"Expert {name!r} has another source base")
            fmt = payload.get("format")
            if payload.get("routing_mode", "competitive") != "competitive":
                raise ValueError("Continuous-fusion experts cannot enter this competitive bank")
            if fmt == CONTEXT_FORMAT:
                kind, dim = "context", payload["context_dim"]
            elif fmt in PLAIN_FORMATS:
                kind, dim = "plain", 0
            else:
                raise ValueError(f"Unsupported expert checkpoint format: {fmt}")
            plugin = _plugin(kind, dim)
            if not _finite(payload["plugin"]):
                raise ValueError(f"Expert {name!r} contains nonfinite parameters")
            plugin.load_state_dict(payload["plugin"], strict=True)
            plugins[name] = plugin.to(device)
            specs[name] = dict(
                kind=kind,
                context_dim=dim,
                source_format=fmt,
                source_checkpoint=_portable(str(Path(path).resolve())),
                source_sha256=file_digest(path),
                actual_Adam_updates=payload.get(
                    "actual_Adam_updates", payload.get("actor_adam_updates")
                ),
                context_contract=_portable(payload.get("context_contract")),
                historical_evaluation=copy.deepcopy((metadata or {}).get(name, {})),
            )
        return cls(pi, args, robot, plugins, specs, base_sha)

    @classmethod
    def load_bundle(cls, path, *, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("format") != FORMAT
            or payload.get("robot") != "g1"
            or payload.get("max_active_experts") != 1
            or payload.get("default_active_expert") is not None
        ):
            raise ValueError("Not a compatible single-active-expert G1 bundle")
        specs = payload["experts"]
        if not _finite(payload["pi"]):
            raise ValueError("Nonfinite Actor state")
        pi, robot = policy_from_args(payload["args"], device)
        plugins = {
            name: _plugin(spec["kind"], spec["context_dim"]).to(device)
            for name, spec in specs.items()
        }
        bank = cls(pi, payload["args"], robot, plugins, specs, payload["base_sha256"])
        bank.pi.load_state_dict(payload["pi"], strict=True)
        bank.public_sha256 = payload["public_sha256"]
        if public_digest(bank.pi) != bank.public_sha256:
            raise ValueError("Bundle public fingerprint mismatch")
        return bank
