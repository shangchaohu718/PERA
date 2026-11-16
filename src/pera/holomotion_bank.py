"""One official frozen HoloMotion Actor containing switchable task FFNs."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf
from safetensors.torch import load_file

from modumotion_g1.domain_moe import (
    install_domains,
    interface_digest,
    load_domain_payload,
    public_digest,
    set_active_domains,
    set_domain_context,
)
from modumotion_g1.official import build_official_actor
from pera.paths import ROOT

FORMAT = "pera_official_holomotion_bank_v1"


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_digest(payload):
    digest = hashlib.sha256()
    for i, layer in enumerate(payload["layers"]):
        for key, value in sorted(layer.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{i}/{key}".encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def context_kwargs(record):
    context = record.get("context")
    return (
        dict(
            context_dim=context["dim"],
            context_schema=context["schema"],
            encoder_sha256=context["encoder_sha256"],
        )
        if context
        else {}
    )


def compose(manifest_path, native_root, base_dir, output, plugins_output):
    """Compose local weights; never upload or copy official models into the release."""
    output, plugins_output = Path(output).resolve(), Path(plugins_output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError(
            "The full checkpoint contains official weights; choose an output outside the source release"
        )
    if output == plugins_output or output.exists() or plugins_output.exists():
        raise ValueError("Use two new output paths; existing checkpoints are preserved")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["base_family"] != "holomotion_official_g1_v1.4":
        raise ValueError("Only the selected official HoloMotion base is supported")
    base = json.loads((ROOT / "resources/holomotion_base.json").read_text())
    base_dir = Path(base_dir)
    for row in base["files"]:
        path = base_dir / Path(row["path"]).relative_to("checkpoints/holomotion_v1.4")
        if file_digest(path) != row["sha256"]:
            raise ValueError(f"Official model hash mismatch: {row['path']}")
    configuration = (base_dir / "config.yaml").read_text()
    state = load_file(str(base_dir / "model_14000/actor/model.safetensors"), device="cpu")
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    experts = {}
    for row in manifest["experts"]:
        path = Path(native_root) / row["source_relative_to_native_root"]
        if file_digest(path) != row["sha256"]:
            raise ValueError(f"Expert source hash mismatch: {row['id']}")
        payload = torch.load(path, weights_only=True, map_location="cpu")
        for key in ["public_sha256", "interface_sha256", "context", "merge_bias"]:
            if payload.get(key) != row.get(key):
                raise ValueError(f"Expert contract mismatch: {row['id']}/{key}")
        if payload["name"] != row["domain"]:
            raise ValueError("Domain name differs from manifest")
        # Preserve all behavior fields and original tensors, omit private run paths.
        payload["provenance"] = {"original_checkpoint_sha256": row["sha256"]}
        experts[row["id"]] = {
            "record": row,
            "payload": payload,
            "tensor_sha256": tensor_digest(payload),
        }
    if len(experts) != len(manifest["experts"]):
        raise ValueError("Duplicate expert IDs")
    bundle = dict(
        format=FORMAT,
        base_family=manifest["base_family"],
        public_sha256=manifest["public_sha256"],
        interface_sha256=manifest["interface_sha256"],
        base_config_yaml=configuration,
        base_config_sha256=hashlib.sha256(configuration.encode()).hexdigest(),
        base_actor_file_sha256=base["files"][1]["sha256"],
        max_active_experts_per_inference=1,
        experts=experts,
        contains_official_actor=False,
    )
    # Strictly reconstruct and validate all embedded domains before saving either file.
    validation = FrozenHoloMotionBank.from_payload(
        {**bundle, "public_state_dict": state, "contains_official_actor": True}, "cpu"
    )
    validation.verify()
    del validation
    for path in [output, plugins_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, plugins_output)
    torch.save({**bundle, "public_state_dict": state, "contains_official_actor": True}, output)
    return dict(
        expert_count=len(experts),
        full_checkpoint_bytes=output.stat().st_size,
        full_checkpoint_sha256=file_digest(output),
        plugin_bank_bytes=plugins_output.stat().st_size,
        plugin_bank_sha256=file_digest(plugins_output),
        official_base_uploaded=False,
    )


class FrozenHoloMotionBank(nn.Module):
    """All task modules are resident; each inference explicitly selects zero or one."""

    @classmethod
    def load(cls, checkpoint, device="cpu", base_dir=None):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not payload.get("contains_official_actor"):
            if base_dir is None:
                raise ValueError(
                    "The expert-only bank needs the downloaded official base directory"
                )
            filename = Path(base_dir) / "model_14000/actor/model.safetensors"
            if file_digest(filename) != payload["base_actor_file_sha256"]:
                raise ValueError("External official Actor hash mismatch")
            state = load_file(str(filename), device="cpu")
            payload["public_state_dict"] = {
                k.removeprefix("_orig_mod."): v for k, v in state.items()
            }
        return cls.from_payload(payload, device)

    @classmethod
    def from_payload(cls, payload, device):
        if (
            payload.get("format") != FORMAT
            or payload.get("base_family") != "holomotion_official_g1_v1.4"
        ):
            raise ValueError("Unsupported HoloMotion bank format")
        expected = json.loads((ROOT / "configs/demo_policy.json").read_text())
        if (
            payload["public_sha256"] != expected["public_sha256"]
            or payload["interface_sha256"] != expected["interface_sha256"]
        ):
            raise ValueError("Bank does not use the selected official public Actor/interface")
        configuration = payload["base_config_yaml"]
        if (
            hashlib.sha256(configuration.encode()).hexdigest()
            != expected["base_files"]["config.yaml"]["sha256"]
        ):
            raise ValueError("Official model configuration hash mismatch")
        if payload.get("max_active_experts_per_inference") != 1:
            raise ValueError("Only one active domain is allowed")
        obj = cls()
        obj.actor = build_official_actor(OmegaConf.create(configuration), device)
        obj.actor.load_state_dict(payload["public_state_dict"], strict=True)
        install_domains(obj.actor)
        obj.public_sha256 = public_digest(obj.actor)
        if (
            obj.public_sha256 != payload["public_sha256"]
            or interface_digest(obj.actor) != payload["interface_sha256"]
        ):
            raise ValueError("Public model or interface digest mismatch")
        obj.records, obj.domains = {}, {}
        for name, entry in payload["experts"].items():
            record, domain = entry["record"], entry["payload"]
            if name != record["id"] or domain["name"] != record["domain"]:
                raise ValueError("Expert identity mismatch")
            for key in ["public_sha256", "interface_sha256", "context", "merge_bias"]:
                if domain.get(key) != record.get(key):
                    raise ValueError(f"Embedded expert contract mismatch: {name}/{key}")
            if tensor_digest(domain) != entry["tensor_sha256"]:
                raise ValueError(f"Embedded expert tensor hash mismatch: {name}")
            obj.domains[name] = load_domain_payload(obj.actor, domain, **context_kwargs(record))
            obj.records[name] = record
        obj.actor.eval().requires_grad_(False)
        obj.expert_names = tuple(obj.domains)
        obj.active_expert = None
        obj.select_expert(None)
        return obj

    def select_expert(self, name):
        if name is not None and not isinstance(name, str):
            raise ValueError("Select one expert ID or None; multiple active experts are forbidden")
        if name is not None and name not in self.domains:
            raise ValueError(f"Unknown expert: {name}")
        set_domain_context(self.actor, None)
        set_active_domains(self.actor, [self.domains[name]] if name is not None else [])
        self.active_expert = name

    @torch.no_grad()
    def forward(self, observations, *, expert=None, context=None):
        self.select_expert(expert)
        try:
            contract = self.records[expert].get("context") if expert is not None else None
            if contract:
                if (
                    context is None
                    or context.ndim != 2
                    or context.shape[-1] != contract["dim"]
                    or not bool(context.isfinite().all())
                ):
                    raise ValueError(f"Missing/invalid frozen context for {expert}")
                set_domain_context(self.actor, {self.domains[expert]: context})
            elif context is not None:
                raise ValueError("This expert/public path does not accept external context")
            return self.actor(observations, mode="inference")
        finally:
            self.select_expert(None)

    def verify(self):
        if public_digest(self.actor) != self.public_sha256:
            raise ValueError("Frozen public Actor changed")
        if any(p.requires_grad for p in self.actor.parameters()):
            raise ValueError("The bank must remain frozen for inference")
        return self.public_sha256
