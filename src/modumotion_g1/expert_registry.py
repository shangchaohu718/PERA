"""Validated shared-Actor expert bundle with explicit exclusive selection."""

import hashlib
import json
from pathlib import Path

import torch

from modumotion_g1.domain_moe import (
    install_domains,
    interface_digest,
    load_domain,
    public_digest,
    set_active_domains,
    set_domain_enabled,
    set_trainable_domain,
)

from pera.paths import ROOT

DEFAULT_REGISTRY = ROOT / "configs/expert_registry.json"


class ExpertRegistry:
    """Runtime switches only; context producers and physical entry remain task-specific.

    Enabling selects exactly one added domain. Disabling removes only that
    domain, leaving any other selected domain untouched. Neither operation
    resets history, changes fixed merge biases, or writes network parameters.
    """

    def __init__(self, actor, specification):
        self.actor = actor
        self.specification = specification
        self.experts = specification["experts"]
        self.names = tuple(self.experts)
        self._aliases = {row["domain"]: name for name, row in self.experts.items()}

    def domain(self, name):
        key = name if name in self.experts else self._aliases.get(name)
        if key is None:
            raise ValueError(f"Unknown registered expert: {name}")
        return self.experts[key]["domain"]

    @property
    def selected(self):
        active = self.actor.actor_module.layers[0].active_domains
        if len(active) > 1:
            raise RuntimeError("Selection was changed outside the exclusive registry")
        return self._aliases[active[0]] if active else None

    def select(self, name=None):
        """Select one added expert, or None for the frozen public path."""
        set_active_domains(self.actor, [] if name is None else [self.domain(name)])

    def set_enabled(self, name, enabled):
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        domain = self.domain(name)
        if enabled:
            set_active_domains(self.actor, [domain])
        else:
            set_domain_enabled(self.actor, domain, False)


def load_expert_registry(actor, path=DEFAULT_REGISTRY):
    """Load the inference bundle into one Actor, initially selecting public only.

    Exact matching plugins already present in the Actor are reused. The public
    release is frozen before any inference, including normalization statistics.
    All manifest/checkpoint contracts are checked before adding any plugins.
    No public weights or attention caches are duplicated or serialized here.
    """
    specification = json.loads(Path(path).read_text())
    if specification.get("format") != "modumotion_g1_expert_registry_v1":
        raise ValueError("Unsupported expert registry")
    if specification.get("default_active_expert") is not None:
        raise ValueError("Expert bundle must start with public only")
    if specification.get("selection_mode") != "exclusive":
        raise ValueError("Expected explicit exclusive expert selection")
    experts = specification["experts"]
    domains = [row["domain"] for row in experts.values()]
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("Registry domains must be nonempty and unique")
    if public_digest(actor) != specification["public_sha256"]:
        raise ValueError("Registry does not match the exact original public Actor")
    prepared = []
    interface = interface_digest(actor)
    for name, row in experts.items():
        plugin_path = Path(row["plugin"])
        plugin_path = plugin_path if plugin_path.is_absolute() else ROOT / plugin_path
        if hashlib.sha256(plugin_path.read_bytes()).hexdigest() != row["plugin_sha256"]:
            raise ValueError(f"Expert checkpoint hash differs: {name}")
        payload = torch.load(plugin_path, map_location="cpu", weights_only=True)
        if (
            payload["name"] != row["domain"]
            or payload["public_sha256"] != specification["public_sha256"]
            or payload["interface_sha256"] != interface
        ):
            raise ValueError(f"Expert name/public/interface differs: {name}")
        if row.get("default_enabled") is not False:
            raise ValueError(f"Expert must default to disabled: {name}")
        ctx = payload.get("context")
        expected = row["context"]
        if ctx is None or (ctx["dim"], ctx["schema"], ctx["encoder_sha256"]) != (
            expected["context_dim"],
            expected["context_schema"],
            expected["encoder_sha256"],
        ):
            raise ValueError(f"Expert context differs: {name}")
        if any(not torch.isfinite(t).all() for layer in payload["layers"] for t in layer.values()):
            raise ValueError(f"Nonfinite expert checkpoint: {name}")
        prepared.append((name, row, plugin_path, payload))
    install_domains(actor)
    registered = set(domains)
    if any(set(block.domain_plugins) - registered for block in actor.actor_module.layers):
        raise ValueError("Actor contains domains outside this bundle")
    for name, row, plugin_path, payload in prepared:
        domain = row["domain"]
        present = [domain in block.domain_plugins for block in actor.actor_module.layers]
        if any(present) and not all(present):
            raise ValueError(f"Expert is missing from some layers: {name}")
        if all(present):
            for block, saved, bias in zip(
                actor.actor_module.layers,
                payload["layers"],
                payload["merge_bias"],
                strict=True,
            ):
                plugin = block.domain_plugins[domain]
                live = plugin.state_dict()
                if (
                    plugin.context_contract != payload["context"]
                    or live.keys() != saved.keys()
                    or any(not torch.equal(live[k].detach().cpu(), saved[k]) for k in saved)
                    or block.merge_bias[domain] != bias
                ):
                    raise ValueError(f"Existing expert differs from registry checkpoint: {name}")
        else:
            load_domain(actor, plugin_path, **row["context"])
    set_trainable_domain(actor, None)
    set_active_domains(actor, [])
    return ExpertRegistry(actor, specification)
