import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pera.context import ContextMoE, ContextPlugin
from pera.routing import Plugin, load_base, policy_from_args, public_digest
from pera.switching import CONTEXT_FORMAT, SwitchedActor, file_digest


@pytest.fixture
def source_models(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(81)
    args = dict(
        robot="g1", hist=10, enc_hidden="32,16", dec_hidden="32,256", amp=None, pi_ref_skip=True
    )
    pi, _ = policy_from_args(args)
    base = tmp_path / "base.pt"
    torch.save(dict(args=args, pi=pi.state_dict(), iteration=0), base)
    digest = public_digest(pi)
    paths = {}
    for name, dim in [("a", 3), ("b", 5), ("martial", 0)]:
        plugin = ContextPlugin(dim) if dim else Plugin()
        with torch.no_grad():
            plugin.gate_up.normal_(0, 0.02)
            plugin.down.normal_(0, 0.02)
            plugin.router.bias.fill_(-0.5)
            plugin.selection_bias.fill_(2.0)
        path = tmp_path / (name + ".pt")
        torch.save(
            dict(
                format=CONTEXT_FORMAT if dim else "blink_g1_martial_seven_ppo_competitive_v1",
                public_sha256=digest,
                base_sha256=file_digest(base),
                context_dim=dim,
                plugin=plugin.state_dict(),
            ),
            path,
        )
        paths[name] = path
    return base, paths


def standalone(base, path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    pi, _, _ = load_base(base, device="cpu")
    if data["format"] == CONTEXT_FORMAT:
        pi.moe.__class__ = ContextMoE
        pi.moe.plugin = ContextPlugin(data["context_dim"])
    pi.moe.plugin.load_state_dict(data["plugin"])
    pi.moe.runtime_context = None
    return pi.eval()


def test_outputs_switch_guards_inactive_experts_and_cache(source_models):
    base, paths = source_models
    bank = SwitchedActor.from_checkpoints(base, paths)
    native, _, _ = load_base(base, device="cpu")
    native.moe.domain_enabled = False
    h, g = torch.randn(4, 10, 608), torch.randn(4, 730)
    with torch.no_grad():
        expected = native(h, g)
        torch.testing.assert_close(bank(h, g), expected, atol=0, rtol=0)
        for name, path in paths.items():
            bank.select_expert(name)
            context = torch.randn(4, bank.context_dim) if bank.context_dim else None
            reference = standalone(base, path)
            reference.moe.runtime_context = context
            torch.testing.assert_close(bank(h, g, context), reference(h, g), atol=0, rtol=0)
            cache = bank.cached_features(g)
            torch.testing.assert_close(
                bank.predict_cached(cache)[0], bank(h, g, context), atol=0, rtol=0
            )
    bank.select_expert("a")
    context = torch.randn(4, 3)
    bank(h, g, context)
    cached = bank.cached_features(g)
    before = public_digest(bank.pi)
    for operation in [
        lambda: bank.set_expert_enabled("b", True),
        lambda: bank.set_switches({"a": True, "b": True}),
        lambda: bank.set_switches({"missing": True}),
        lambda: bank.set_switches({"b": "true"}),
    ]:
        with pytest.raises((ValueError, TypeError)):
            operation()
        assert bank.active_expert == "a"
    bank.select_expert("b")
    with pytest.raises(ValueError, match="predate"):
        bank.predict_cached(cached)
    with pytest.raises(ValueError, match="fresh"):
        bank(h, g)
    with pytest.raises(ValueError, match="mismatch"):
        bank(h, g, context)
    bank.set_switches({})

    # A single active expert must never execute the other installed experts.
    def forbidden(*args, **kwargs):
        raise AssertionError("A closed expert was evaluated")

    bank.select_expert("a")
    bank.pi.moe.plugins["b"].score = forbidden
    bank.pi.moe.plugins["b"].forward = forbidden
    bank.pi.moe.plugins["martial"].score = forbidden
    bank.pi.moe.plugins["martial"].forward = forbidden
    bank(h, g, context)
    bank.select_expert(None)
    # Poison every closed router and FFN: the all-off public path still executes exactly.
    for plugin in bank.pi.moe.plugins.values():
        plugin.score = forbidden
        plugin.forward = forbidden
    torch.testing.assert_close(bank(h, g), expected, atol=0, rtol=0)
    assert before == public_digest(bank.pi)
    assert all(not p.requires_grad for p in bank.parameters())
    assert not torch.cuda.is_initialized()


def test_bundle_is_self_contained_and_all_off_on_load(source_models, tmp_path, monkeypatch):
    base, paths = source_models
    bank = SwitchedActor.from_checkpoints(base, paths)
    h, g, ctx = torch.randn(2, 10, 608), torch.randn(2, 730), torch.randn(2, 3)
    bank.select_expert("a")
    expected = bank(h, g, ctx)
    path = tmp_path / "combined.pt"
    bank.save_bundle(path)
    for p in [base, *paths.values()]:
        p.unlink()
    restored = SwitchedActor.load_bundle(path)
    assert restored.active_expert is None and not any(restored.switches.values())
    assert len(restored.expert_names) == 3
    restored.select_expert("a")
    torch.testing.assert_close(restored(h, g, ctx), expected, atol=0, rtol=0)
    assert len([n for n, _ in restored.named_modules() if n.endswith("router_embed")]) == 1
    assert not any(key.startswith("moe.plugin.") for key in restored.pi.state_dict())
    with pytest.raises(FileExistsError):
        restored.save_bundle(path)
    payload = torch.load(path, weights_only=True)
    payload["max_active_experts"] = 2
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        SwitchedActor.load_bundle(tmp_path / "bad.pt")


def test_rejects_wrong_base_fusion_and_invalid_shapes(source_models, tmp_path):
    base, paths = source_models
    original = torch.load(paths["a"], weights_only=True)
    for key, value in [
        ("public_sha256", "wrong"),
        ("base_sha256", "wrong"),
        ("routing_mode", "fusion"),
        ("context_dim", 4),
    ]:
        bad = {**original, key: value}
        path = tmp_path / "bad.pt"
        torch.save(bad, path)
        with pytest.raises((ValueError, RuntimeError)):
            SwitchedActor.from_checkpoints(base, {"bad": path})


def test_reference_bridge_shares_actor_and_preserves_history(source_models, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "reproduction"))
    from experiments.g1_competitive_suite31_v1.model import Actor, CAPTURE

    base, paths = source_models
    bank = SwitchedActor.from_checkpoints(base, paths)
    bank.select_expert("a")
    bridge = bank.reference_actor()
    original = Actor(base, 3, "a", device="cpu")
    original.pi.moe.plugin.load_state_dict(torch.load(paths["a"], weights_only=True)["plugin"])
    assert bridge.pi is bank.pi
    ref = torch.zeros(2, 13, 36)
    ref[..., 2], ref[..., 3] = 1.0, 1.0
    raw = SimpleNamespace(
        reference_qpos=ref,
        robot_root_quat_wxyz=torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
        robot_dof_pos=torch.zeros(2, 29),
        robot_dof_vel=torch.zeros(2, 29),
        robot_root_angvel_local=torch.zeros(2, 3),
        last_action=torch.zeros(2, 29),
    )
    saved = copy.copy(CAPTURE)
    try:
        CAPTURE.update(inputs=raw, version=1, positions=None)
        context = torch.randn(2, 3)
        original.context = bridge.context = context
        with torch.no_grad():
            torch.testing.assert_close(bridge()["actions"], original()["actions"], atol=0, rtol=0)
        history = bridge.history
        bank.select_expert("b")
        assert bridge.context is None and bridge.history is history
        bridge._dispatch_bank_control("set_domain_context", {"b": torch.randn(2, 5)})
        bridge()
        assert bridge.history is history
        with pytest.raises(ValueError):
            bridge._dispatch_bank_control("set_active_domains", ["a", "b"])
        assert bank.active_expert == "b"
        bridge._dispatch_bank_control("set_active_domains", [])
        assert bank.active_expert is None
    finally:
        CAPTURE.clear()
        CAPTURE.update(saved)


def test_multi_robot_dispatch_histories_and_contexts_are_independent(source_models):
    from pera.multi_robot import MultiRobotController

    base, paths = source_models
    bank = SwitchedActor.from_checkpoints(base, paths)
    controller = MultiRobotController(bank, ["a", "b", "martial", None])
    observation, goal = torch.randn(4, 608), torch.randn(4, 730)
    contexts = [torch.randn(3), torch.randn(5), None, None]
    result = controller.infer(observation, goal, contexts)
    assert controller.actor is bank and bank.active_expert is None
    for i, name in enumerate(controller.experts):
        bank.select_expert(name)
        context = contexts[i][None] if contexts[i] is not None else None
        torch.testing.assert_close(
            result[i : i + 1],
            bank(controller.history[i : i + 1], goal[i : i + 1], context),
            atol=0,
            rtol=0,
        )
    previous = controller.history.clone()
    controller.reset([1])
    newer = observation + 1
    controller.infer(newer, goal, contexts)
    torch.testing.assert_close(controller.history[0, :-1], previous[0, 1:])
    torch.testing.assert_close(controller.history[1], newer[1, None].expand(10, -1))
    assert bank.active_expert is None
    saved = controller.history.clone()
    with pytest.raises(ValueError, match="context"):
        controller.infer(newer, goal, [torch.randn(5), contexts[1], None, None])
    torch.testing.assert_close(controller.history, saved)


def test_multi_robot_groups_identical_experts_without_copying_actor(source_models):
    from pera.multi_robot import MultiRobotController

    base, paths = source_models
    bank = SwitchedActor.from_checkpoints(base, paths)
    controller = MultiRobotController(bank, ["a", "a", "b"])
    observations, goal = torch.randn(3, 608), torch.randn(3, 730)
    context = [torch.randn(3), torch.randn(3), torch.randn(5)]
    output = controller.infer(observations, goal, context)
    assert [row["robots"] for row in controller.last_dispatch] == [[0, 1], [2]]
    bank.select_expert("a")
    expected = bank(controller.history[:2], goal[:2], torch.stack(context[:2]))
    torch.testing.assert_close(output[:2], expected, atol=0, rtol=0)
    assert len([n for n, _ in controller.actor.pi.named_modules() if n == "moe.router_embed"]) == 1


def test_nonfinite_expert_is_reported_before_history_commit(source_models):
    from pera.multi_robot import MultiRobotController, NonfinitePolicyAction

    base, paths = source_models
    bank = SwitchedActor.from_checkpoints(base, paths)
    controller = MultiRobotController(bank, ["a", "b"])
    obs, goal = torch.randn(2, 608), torch.randn(2, 730)
    contexts = [torch.randn(3), torch.randn(5)]
    # Both routers are admitted by the fixture. Poison only expert b.
    bank.pi.moe.plugins["b"].forward = lambda x: x.new_full((len(x), 256), float("nan"))
    with pytest.raises(NonfinitePolicyAction) as caught:
        controller.infer(obs, goal, contexts)
    assert caught.value.robot_ids == (1,)
    assert controller.history is None and controller.pending_reset.all()
    assert bank.active_expert is None
    controller.experts = ("a", None)
    result = controller.infer(obs, goal, [contexts[0], None])
    assert result.isfinite().all()
    bank.select_expert(None)
    torch.testing.assert_close(
        result[1:2], bank(controller.history[1:2], goal[1:2]), atol=0, rtol=0
    )
