"""CPU-only checks of the actual routing, replay and frozen-parameter contracts."""

import copy
import json
import torch
from torch import nn
from blinkmimic import networks
from blinkmimic.moe_policy import ReferenceMoE
from .routing import CompetitiveMoE, Plugin, public_digest


def verify():
    torch.set_num_threads(2)
    torch.manual_seed(71)
    networks.AMP_DTYPE = None
    native = ReferenceMoE().eval().requires_grad_(False)
    policy = copy.deepcopy(native)
    policy.__class__ = CompetitiveMoE
    policy.plugin = Plugin()
    policy.domain_enabled = False
    x = torch.randn(32, 256)
    r = torch.randn(32, 509)
    public, stats = native(x, r)
    off, _ = policy(x, r)
    assert torch.equal(public, off), "Disabled expert changed the old path"
    policy.domain_enabled = True
    with torch.no_grad():
        policy.plugin.router.bias.fill_(-1e6)
    rejected, _ = policy(x, r)
    assert not policy.last_selected.any() and torch.equal(public, rejected)
    with torch.no_grad():
        policy.plugin.router.bias.fill_(-0.5)
        policy.plugin.selection_bias.fill_(1e6)
        policy.plugin.gate_up.normal_(0, 0.03)
        policy.plugin.down.normal_(0, 0.03)
    wrapper = nn.Module()
    wrapper.moe = policy
    before = public_digest(wrapper)
    head = nn.Linear(256, 29).requires_grad_(False)
    opt = torch.optim.AdamW(policy.plugin.parameters(), lr=1e-3, weight_decay=0.0)
    full, _ = policy(x, r)
    f = dict(policy.last_features)
    f["skip"] = torch.zeros(32, 29)
    predicted = head(x + full)
    cached, *_ = policy.predict(f, head)
    torch.testing.assert_close(predicted, cached, rtol=0, atol=0)
    # Independent nine-candidate Top2 reference, selecting on corrected scores
    # and mixing on raw logits, with exactly one shared FFN contribution.
    xn = native.norm(x)
    logits = native.logits(r)
    hidden = native.router_embed(r)
    raw_new = policy.plugin.score(hidden)
    logits9 = torch.cat((logits, raw_new), -1)
    correction = torch.cat((native.selection_bias, policy.plugin.selection_bias.reshape(1)))
    ids = (logits9 + correction).topk(2, -1).indices
    g, u = torch.einsum("bd,edi->bei", xn, native.gate_up).chunk(2, -1)
    outputs = torch.einsum("beh,ehd->bed", torch.nn.functional.silu(g) * u, native.down)
    outputs = torch.cat((outputs, policy.plugin(xn)[:, None]), 1)
    picked = outputs.gather(1, ids[..., None].expand(-1, -1, 256))
    g, u = native.shared_up(xn).chunk(2, -1)
    expected = native.shared_down(torch.nn.functional.silu(g) * u) + (
        picked * logits9.gather(1, ids).softmax(-1)[..., None]
    ).sum(1)
    torch.testing.assert_close(full, expected, rtol=1e-5, atol=2e-6)
    target = torch.randn_like(cached)
    loss = (cached - target).square().mean()
    grads = torch.autograd.grad(loss, tuple(policy.plugin.parameters()))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    initial = {k: p.detach().clone() for k, p in policy.plugin.named_parameters()}
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out, *_ = policy.predict(f, head)
        (out - target).square().mean().backward()
        opt.step()
    assert public_digest(wrapper) == before, "Public tensors changed"
    assert all(
        not torch.equal(v, dict(policy.plugin.named_parameters())[k]) for k, v in initial.items()
    )
    assert sorted({int(v["step"]) for v in opt.state.values()}) == [3]
    assert all(p.grad is None for n, p in policy.named_parameters() if not n.startswith("plugin."))
    policy.domain_enabled = False
    cached_off, *_ = policy.predict(f, head)
    torch.testing.assert_close(cached_off, head(x + public), atol=0, rtol=0)
    assert not torch.cuda.is_initialized(), "CPU check initialized CUDA"
    return dict(
        disabled_exact_public=True,
        rejected_exact_public=True,
        agrees_with_full_nine_way_top2=True,
        cached_full_forward_exact=True,
        four_new_tensors_updated=True,
        actual_adam_updates=3,
        public_unchanged=True,
        CUDA_initialized=False,
    )


def main():
    print(json.dumps(verify(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
