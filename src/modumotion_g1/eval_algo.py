"""Add domains to the official evaluator after strict public loading."""

import os
from pathlib import Path

from holomotion.src.algo.ppo_tf import PPOTF

from modumotion_g1.domain_moe import (
    install_domains,
    load_domain,
    set_active_domains,
)


class DomainEvalPPO(PPOTF):
    def load(self, ckpt_path):
        if not self.is_offline_eval:
            raise ValueError("This adapter is for simulation evaluation only")
        result = super().load(ckpt_path)
        actor = self.accelerator.unwrap_model(self.actor)
        install_domains(actor)
        paths = [
            Path(p) for p in os.environ.get("MODUMOTION_DOMAIN_PATHS", "").split(os.pathsep) if p
        ]
        names = [load_domain(actor, p) for p in paths]
        override = os.environ.get("MODUMOTION_MERGE_BIAS")
        biases = None if override is None else dict.fromkeys(names, float(override))
        set_active_domains(actor, names, biases)
        return result
