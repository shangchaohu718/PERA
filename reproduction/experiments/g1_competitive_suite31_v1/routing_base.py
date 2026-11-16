"""Compatibility with original experiment checkpoints; routing is shared."""

from pera.routing import CompetitiveMoE, Plugin, features, public_digest  # noqa: F401 -- compatibility exports
from pera.routing import load_base as _load, initialize as _initialize

FORMAT = "blink_g1_eight_plus_one_competitive_v1"


def load_base(path, device="cuda:0", domain=None):
    return _load(path, device, domain, expected_format=FORMAT)


def initialize(pi, bank, old_ids, target=0.95):
    return _initialize(pi, bank, old_ids, target)
