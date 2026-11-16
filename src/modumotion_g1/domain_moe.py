"""Continuous anchor routing in the native G1 MoE hidden FFN space.

The public 1024-expert Top-2 and its state-dict keys stay intact. This is a
HoloMotion adaptation of independent anchored expert training, not the LM
architecture or domain-embedding router from the FlexOlmo paper.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
from holomotion.src.modules.agent_modules import PPOTFRefRouterActor
from holomotion.src.modules.network_modules import GroupedMoEBlock
from torch import nn
from torch.nn import functional as F


def _context_contract(context_dim, context_schema=None, encoder_sha256=None):
    """Describe an external frozen encoder without registering its weights."""
    if isinstance(context_dim, bool) or not isinstance(context_dim, int) or context_dim < 0:
        raise ValueError("context_dim must be a nonnegative integer")
    if context_dim == 0:
        if context_schema is not None or encoder_sha256 is not None:
            raise ValueError("A context contract requires positive context_dim")
        return None
    if not isinstance(context_schema, dict) or not context_schema:
        raise ValueError("Provide a nonempty JSON context schema")
    if (
        not isinstance(encoder_sha256, str)
        or len(encoder_sha256) != 64
        or any(c not in "0123456789abcdef" for c in encoder_sha256)
    ):
        raise ValueError("Provide the frozen encoder SHA256 in lowercase hex")
    try:
        schema = json.loads(json.dumps(context_schema, allow_nan=False))
        contract = {
            "dim": context_dim,
            "schema": schema,
            "encoder_sha256": encoder_sha256,
        }
        serialized = json.dumps(contract, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Context schema must contain finite JSON data") from error
    contract["sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    return contract


def public_digest(actor):
    """Hash public parameters and persistent normalization/router buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        if ".domain_plugins." in name:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def interface_digest(actor):
    cfg = actor.actor_module.module_config_dict
    # Fingerprint behavior, not PPO-only auxiliary-loss flags or redundant
    # input_dim_override fields inserted by the official simulation runner.
    semantic_keys = (
        "type",
        "num_fine_experts",
        "num_shared_experts",
        "top_k",
        "routing_score_fn",
        "routing_scale",
        "use_dynamic_bias",
        "d_model",
        "n_heads",
        "n_kv_heads",
        "n_layers",
        "max_ctx_len",
        "rope_max_seq_len",
        "dense_layer_at_first",
        "ff_mult",
        "ff_mult_dense",
        "use_gated_attn",
        "attn_dropout",
        "mlp_dropout",
        "obs_norm",
        "router_obs_terms",
    )
    contract = {
        "schema": actor.assembler.schema_config,
        "action_dim": actor.actor_module.output_dim,
        "input_dim": actor.assembler.output_dim,
        "router_indices": actor.router_feature_indices,
        "network": {k: cfg[k] for k in semantic_keys if k in cfg},
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


class DomainFFN(nn.Module):
    def __init__(
        self,
        block,
        source_expert=0,
        initial_bias=-2.0,
        *,
        context_dim=0,
        context_schema=None,
        encoder_sha256=None,
    ):
        super().__init__()
        self.context_contract = _context_contract(context_dim, context_schema, encoder_sha256)
        self.context_dim = context_dim
        if not 0 <= source_expert < block.num_fine_experts:
            raise ValueError("Invalid public expert initialization index")
        gate_up = block.gate_up_proj[source_expert].detach().clone()
        if context_dim:
            gate_up = torch.cat((gate_up, gate_up.new_zeros(context_dim, gate_up.shape[1])))
        self.gate_up = nn.Parameter(gate_up)
        self.down = nn.Parameter(block.down_proj[source_expert].detach().clone())
        self.router = nn.Linear(block.d_model + context_dim, 1, bias=True).to(block.router.weight)
        with torch.no_grad():
            self.router.weight.zero_()
            self.router.weight[:, : block.d_model].copy_(
                block.router.weight[source_expert : source_expert + 1]
            )
            self.router.bias.fill_(initial_bias)

    def _with_context(self, x, context):
        if not self.context_dim:
            return x
        if not isinstance(context, torch.Tensor):
            raise ValueError("This domain requires an external context tensor")
        if not context.is_floating_point():
            raise ValueError("Domain context must be a floating point tensor")
        if context.device != x.device:
            raise ValueError("Domain context and hidden features must share a device")
        # A single-step [B,C] input can accompany [B,1,D]. Never silently
        # broadcast a current image over multiple sequence time steps.
        if x.ndim == 3 and x.shape[1] == 1 and context.ndim == 2:
            context = context.unsqueeze(1)
        if context.shape != (*x.shape[:-1], self.context_dim):
            raise ValueError(
                "Domain context shape must match hidden batch/time: "
                f"expected {(*x.shape[:-1], self.context_dim)}, "
                f"got {tuple(context.shape)}"
            )
        return torch.cat((x, context.to(dtype=x.dtype)), dim=-1)

    def route(self, router_x, context=None):
        return self.router(self._with_context(router_x, context))

    def trainable_parameters(self):
        """The only four tensors an Actor optimizer may update."""
        return self.gate_up, self.down, self.router.weight, self.router.bias

    def forward(self, x, context=None):
        x = self._with_context(x, context)
        gate, up = F.linear(x, self.gate_up.T).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, self.down.T)


class AnchorDomainBlock(GroupedMoEBlock):
    def accumulate_routing_stats(self, topk_idx):
        pass

    def apply_bias_update_from_counts(self):
        pass

    def _update_routed_expert_stats_and_floor_loss(self, **kwargs):
        zero = kwargs["dense_distribution"].new_zeros(())
        self.last_moe_load_balance_loss = zero
        self.last_inactive_expert_margin_to_topk_loss = zero
        self.last_selected_expert_margin_to_unselected_loss = zero
        return zero

    def compute_moe_ffn(self, x, router_x=None, *, return_routing_debug=False):
        if torch.onnx.is_in_onnx_export():
            raise NotImplementedError("Domain FFN ONNX export has not been validated")
        native, indices, old_logits = super().compute_moe_ffn(
            x, router_x, return_routing_debug=True
        )
        enabled = self.active_domains
        if not enabled:
            self.last_group_weights = x.new_ones((*x.shape[:-1], 1))
            return (native, indices, old_logits) if return_routing_debug else native
        if router_x is None or router_x.shape != x.shape:
            raise ValueError("Domain routing requires native reference features")
        anchor = old_logits.float().logsumexp(-1, keepdim=True) - math.log(self.num_fine_experts)
        scores = [
            self.domain_plugins[name].route(router_x, self.domain_contexts.get(name)).float()
            - anchor
            + self.merge_bias[name]
            for name in enabled
        ]
        if len(enabled) == 1:
            weight = scores[0].sigmoid()
            weights = torch.cat((1 - weight, weight), dim=-1)
        else:
            weights = torch.cat([torch.zeros_like(anchor), *scores], dim=-1).softmax(-1)
        # Same operation order as anchor_pair_moe.py: shared FFN is inside each
        # group, and the normalized weights include its contribution once.
        shared = self.shared_experts(x)
        feature_sink = getattr(self, "dagger_feature_sink", None)
        if feature_sink is not None:
            feature_sink(x, router_x, native, anchor, shared)
        output = weights[..., :1].to(x.dtype) * native
        for i, name in enumerate(enabled, 1):
            domain = self.mlp_dropout(
                shared
                + self.routing_scale * self.domain_plugins[name](x, self.domain_contexts.get(name))
            )
            output = output + weights[..., i : i + 1].to(x.dtype) * domain
        self.last_group_weights = weights.detach()
        # Native diagnostics retain exactly 1024 columns and native Top-2 IDs.
        return (output, indices, old_logits) if return_routing_debug else output


class FrozenPublicActor(PPOTFRefRouterActor):
    def train(self, mode=True):
        # Evaluation mode does not disable autograd. It freezes normalizers,
        # dropout and the public trunk while domain parameters still learn.
        super().train(False)
        return self

    def forward(self, *args, **kwargs):
        kwargs["update_obs_norm"] = False
        return super().forward(*args, **kwargs)


def install_domains(actor):
    """Specialize blocks in place, preserving trunk, attention and cache."""
    if isinstance(actor, FrozenPublicActor):
        return actor
    if type(actor) is not PPOTFRefRouterActor:
        raise TypeError("Expected original PPOTFRefRouterActor")
    if any(type(b) is not GroupedMoEBlock for b in actor.actor_module.layers):
        raise ValueError("Expected only native grouped MoE layers")
    actor.__class__ = FrozenPublicActor
    for block in actor.actor_module.layers:
        block.__class__ = AnchorDomainBlock
        block.domain_plugins = nn.ModuleDict()
        # Ordinary runtime attributes, never buffers/parameters: retaining the
        # original Tensor is required for world-model BPTT through context.
        block.domain_contexts = {}
        block.active_domains = ()
        block.merge_bias = {}
        block.last_group_weights = None
        block.freeze_router = True
    actor.requires_grad_(False)
    actor.eval()
    actor.public_sha256 = public_digest(actor)
    actor.interface_sha256 = interface_digest(actor)
    return actor


def add_domain(
    actor,
    name,
    source_expert=0,
    initial_bias=-2.0,
    *,
    context_dim=0,
    context_schema=None,
    encoder_sha256=None,
):
    if getattr(actor, "domain_mode", None) == "soft_residual":
        raise ValueError("Cannot add competitive domains to residual mode")
    if not name.isidentifier():
        raise ValueError("Domain name must be a Python identifier")
    if not math.isfinite(initial_bias):
        raise ValueError("Initial router bias must be finite")
    _context_contract(context_dim, context_schema, encoder_sha256)
    install_domains(actor)
    if any(name in b.domain_plugins for b in actor.actor_module.layers):
        raise ValueError(f"Duplicate domain: {name}")
    for block in actor.actor_module.layers:
        block.domain_plugins[name] = DomainFFN(
            block,
            source_expert,
            initial_bias,
            context_dim=context_dim,
            context_schema=context_schema,
            encoder_sha256=encoder_sha256,
        )
        block.merge_bias[name] = 0.0
    set_trainable_domain(actor, name)
    set_active_domains(actor, [name])


def set_trainable_domain(actor, name=None):
    if name is not None and any(name not in b.domain_plugins for b in actor.actor_module.layers):
        raise ValueError(f"Unknown domain: {name}")
    actor.requires_grad_(False)
    for parameter in actor.parameters():
        parameter.grad = None
    if name is not None:
        for block in actor.actor_module.layers:
            for parameter in block.domain_plugins[name].trainable_parameters():
                parameter.requires_grad_(True)
    actor.eval()


def set_domain_context(actor, context):
    """Set differentiable external features; ``None`` clears retained graphs.

    A tensor is shared by all contextual domains. A dict maps domain names to
    tensors; omitted domains are cleared. Pass [B,C] for a single step or
    [B,T,C] for a sequence. Call this for every imagination state/sequence and
    clear it after the objective has been formed to release runtime references.
    The external encoder remains the caller's responsibility and is not part
    of the public Actor or its optimizer.
    """
    install_domains(actor)
    names = set(actor.actor_module.layers[0].domain_plugins)
    if context is None:
        contexts = {}
    elif isinstance(context, torch.Tensor):
        contexts = {
            name: context
            for name in names
            if actor.actor_module.layers[0].domain_plugins[name].context_dim
        }
    elif isinstance(context, dict):
        if not set(context) <= names:
            raise ValueError("Context provided for an unknown domain")
        contexts = dict(context)
    else:
        raise ValueError("Domain context must be a tensor, domain dict or None")
    for block in actor.actor_module.layers:
        for name, value in contexts.items():
            dim = block.domain_plugins[name].context_dim
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.ndim not in (2, 3)
                or value.shape[-1] != dim
            ):
                raise ValueError(
                    f"Domain {name!r} context must be floating [B,{dim}] or [B,T,{dim}]"
                )
    for block in actor.actor_module.layers:
        block.domain_contexts = dict(contexts)


def set_active_domains(actor, names, merge_bias=None):
    if getattr(actor, "domain_mode", None) == "soft_residual" and merge_bias is not None:
        raise ValueError("Residual mode does not use competitive merge bias")
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate enabled domains")
    for block in actor.actor_module.layers:
        if not set(names) <= set(block.domain_plugins):
            raise ValueError("Unknown enabled domain")
    if merge_bias is not None:
        if set(merge_bias) != set(names) or any(
            not math.isfinite(v) or v > 0 for v in merge_bias.values()
        ):
            raise ValueError("Provide a finite nonpositive bias for each enabled domain")
    for block in actor.actor_module.layers:
        block.active_domains = names
        if merge_bias is not None:
            block.merge_bias.update(merge_bias)


def set_domain_enabled(actor, name, enabled):
    """Manually toggle one loaded domain without changing weights or caches.

    Disabled domains are omitted before router/FFN evaluation, so their
    contribution is exactly zero. Other active domains and merge biases are
    preserved. This runtime choice must be reapplied after loading a plugin.
    """
    if not isinstance(enabled, bool):
        raise TypeError("Domain enabled must be a bool")
    if any(name not in block.domain_plugins for block in actor.actor_module.layers):
        raise ValueError(f"Unknown domain: {name}")
    names = list(actor.actor_module.layers[0].active_domains)
    if enabled and name not in names:
        names.append(name)
    elif not enabled and name in names:
        names.remove(name)
    set_active_domains(actor, names)


def domain_parameters(actor):
    params = [p for p in actor.parameters() if p.requires_grad]
    allowed = {
        id(p)
        for block in actor.actor_module.layers
        for plugin in block.domain_plugins.values()
        for p in plugin.trainable_parameters()
    }
    if not params or any(id(p) not in allowed for p in params):
        raise ValueError("Actor optimizer must contain only domain FFN/router parameters")
    return params


def save_domain(actor, name, path, provenance=None):
    if getattr(actor, "domain_mode", None) == "soft_residual":
        raise ValueError("Use save_soft_residual for a residual Actor")
    if public_digest(actor) != actor.public_sha256:
        raise ValueError("Frozen public Actor changed")
    contracts = [b.domain_plugins[name].context_contract for b in actor.actor_module.layers]
    if any(contract != contracts[0] for contract in contracts):
        raise ValueError("Domain context contracts differ between layers")
    payload = {
        "format": (
            "modumotion_g1_anchor_domain_v2"
            if contracts[0] is not None
            else "modumotion_g1_anchor_domain_v1"
        ),
        "name": name,
        "public_sha256": actor.public_sha256,
        "interface_sha256": actor.interface_sha256,
        "provenance": provenance or {},
        "merge_bias": [b.merge_bias[name] for b in actor.actor_module.layers],
        "layers": [
            {k: v.detach().cpu().clone() for k, v in b.domain_plugins[name].state_dict().items()}
            for b in actor.actor_module.layers
        ],
    }
    if contracts[0] is not None:
        payload["context"] = json.loads(json.dumps(contracts[0]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_domain(
    actor,
    path,
    *,
    context_dim=None,
    context_schema=None,
    encoder_sha256=None,
):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return load_domain_payload(
        actor,
        payload,
        context_dim=context_dim,
        context_schema=context_schema,
        encoder_sha256=encoder_sha256,
    )


def load_domain_payload(
    actor,
    payload,
    *,
    context_dim=None,
    context_schema=None,
    encoder_sha256=None,
):
    """Load an embedded domain using the same strict native checkpoint checks."""
    if getattr(actor, "domain_mode", None) == "soft_residual":
        raise ValueError("Cannot load competitive domains into residual mode")
    install_domains(actor)
    version = payload["format"]
    if version not in (
        "modumotion_g1_anchor_domain_v1",
        "modumotion_g1_anchor_domain_v2",
    ):
        raise ValueError("Unsupported domain checkpoint")
    if version.endswith("v2"):
        saved_contract = payload.get("context")
        if not isinstance(saved_contract, dict) or set(saved_contract) != {
            "dim",
            "schema",
            "encoder_sha256",
            "sha256",
        }:
            raise ValueError("Missing or invalid saved context contract")
        verified = _context_contract(
            saved_contract["dim"],
            saved_contract["schema"],
            saved_contract["encoder_sha256"],
        )
        if verified is None or saved_contract != verified:
            raise ValueError("Saved context contract SHA256 differs")
        if context_dim is None or context_schema is None or encoder_sha256 is None:
            raise ValueError(
                "Loading v2 requires expected context_dim, context_schema and encoder_sha256"
            )
        expected = _context_contract(context_dim, context_schema, encoder_sha256)
        if expected != saved_contract:
            raise ValueError("Domain context/encoder contract differs")
    else:
        if context_dim not in (None, 0) or context_schema is not None or encoder_sha256 is not None:
            raise ValueError("A v1 domain does not accept a context contract")
        context_dim = 0
    if (
        payload["public_sha256"] != actor.public_sha256
        or public_digest(actor) != actor.public_sha256
    ):
        raise ValueError("Domain does not match the exact frozen public Actor")
    if payload["interface_sha256"] != actor.interface_sha256:
        raise ValueError("Domain observation/action/network contract differs")
    if len(payload["layers"]) != len(actor.actor_module.layers):
        raise ValueError("Domain layer count differs")
    merge_bias = payload.get("merge_bias", [0.0] * len(payload["layers"]))
    if len(merge_bias) != len(payload["layers"]) or any(
        not math.isfinite(v) or v > 0 for v in merge_bias
    ):
        raise ValueError("Domain merge bias must be finite and nonpositive")
    name = payload["name"]
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Invalid saved domain name")
    if any(name in b.domain_plugins for b in actor.actor_module.layers):
        raise ValueError(f"Duplicate domain: {name}")
    # Validate all tensor keys/shapes before mutating the live Actor.
    for block, state in zip(actor.actor_module.layers, payload["layers"], strict=True):
        candidate = DomainFFN(
            block,
            context_dim=context_dim,
            context_schema=context_schema,
            encoder_sha256=encoder_sha256,
        )
        candidate.load_state_dict(state, strict=True)
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Nonfinite domain weights")
    add_domain(
        actor,
        name,
        context_dim=context_dim,
        context_schema=context_schema,
        encoder_sha256=encoder_sha256,
    )
    for block, state, bias in zip(
        actor.actor_module.layers, payload["layers"], merge_bias, strict=True
    ):
        block.domain_plugins[name].load_state_dict(state, strict=True)
        block.merge_bias[name] = float(bias)
    set_trainable_domain(actor, None)
    return name
