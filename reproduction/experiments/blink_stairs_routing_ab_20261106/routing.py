"""Historical A/B checkpoint adapter; no duplicate routing implementation."""

from pera.routing import Plugin, features, public_digest
from pera.routing import load_base as _load
from pera.fusion import FusionMoE as CompetitiveMoE, initialize

FORMAT = "blink_g1_stairs_routing_ab_v1"


def load_base(path, device="cuda:0", domain=None, mode="competitive"):
    pi, cfg, robot = _load(path, device, domain, mode=mode, expected_format=FORMAT)
    pi.moe.__class__ = CompetitiveMoE
    return pi, cfg, robot
