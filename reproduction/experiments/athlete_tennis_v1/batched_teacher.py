"""Frozen batched reconstruction of the official ONNX-exported teacher.

All arrays are read from the local release ONNX. The layout and Transformer
are the upstream _OnnxIntentTransformerActor; no teacher fitting is performed.
"""

import onnx
from onnx import numpy_helper
from experiments.athlete_tennis_v1.distill import POLICY, torch, np
from torch import nn


class BatchedTeacher(nn.Module):
    def __init__(self, device="cuda:0"):
        super().__init__()
        graph = onnx.load(str(POLICY)).graph
        arrays = {
            x.name: torch.from_numpy(numpy_helper.to_array(x).copy()) for x in graph.initializer
        }
        self.register_buffer("student_position", arrays["student_position"])
        for kind, mean, denominator in (
            ("obs", "obs_normalizer._mean", "onnx::Div_459"),
            ("state", "state_history_normalizer._mean", "onnx::Div_476"),
            ("ball", "ball_history_normalizer._mean", "onnx::Div_478"),
        ):
            self.register_buffer(kind + "_mean", arrays[mean])
            self.register_buffer(kind + "_denominator", arrays[denominator])
        self.state_input = nn.Linear(93, 128)
        self.ball_input = nn.Linear(6, 128)
        layer = nn.TransformerEncoderLayer(
            128, 4, 256, dropout=0.0, activation="gelu", batch_first=True
        )
        self.student_temporal_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.student_intent_norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(
            nn.Linear(264, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 29),
        )
        state = self.state_dict()
        transposed = {
            "state_input.weight": "onnx::MatMul_477",
            "ball_input.weight": "onnx::MatMul_479",
        }
        for i, names in enumerate((("480", "519", "520"), ("521", "560", "561"))):
            prefix = f"student_temporal_encoder.layers.{i}."
            for target, source in zip(
                ("self_attn.in_proj_weight", "linear1.weight", "linear2.weight"), names
            ):
                transposed[prefix + target] = "onnx::MatMul_" + source
        buffers = {
            "student_position",
            "obs_mean",
            "obs_denominator",
            "state_mean",
            "state_denominator",
            "ball_mean",
            "ball_denominator",
        }
        for name in state:
            if name in transposed:
                state[name] = arrays[transposed[name]].T.contiguous()
            elif name not in buffers:
                state[name] = arrays[name]
        self.load_state_dict(state, strict=True)
        self.to(device).eval().requires_grad_(False)

    def forward(self, observation):
        assert observation.shape[-1] == 631
        student = (observation[:, :136] - self.obs_mean) / self.obs_denominator
        states = observation[:, 136:601].reshape(-1, 5, 93)
        ball = torch.cat(
            (observation[:, 601:616].reshape(-1, 5, 3), observation[:, 616:631].reshape(-1, 5, 3)),
            dim=-1,
        )
        tokens = (
            self.state_input((states - self.state_mean) / self.state_denominator)
            + self.student_position
        )
        tokens = tokens + self.ball_input((ball - self.ball_mean) / self.ball_denominator)
        latent = self.student_intent_norm(self.student_temporal_encoder(tokens)[:, -1])
        return self.mlp(torch.cat((student, latent), dim=-1))

    @torch.no_grad()
    def verify(self, original, observations, atol=2e-4):
        original_actions = np.stack([original(x) for x in observations])
        device = self.student_position.device
        prediction = (
            self(torch.tensor(observations, dtype=torch.float32, device=device)).cpu().numpy()
        )
        error = np.abs(prediction - original_actions)
        assert float(error.max()) < atol, float(error.max())
        assert not any(p.requires_grad for p in self.parameters())
        return {
            "samples": len(observations),
            "max_abs_action_error": float(error.max()),
            "RMSE_action_error": float(np.sqrt(np.mean(error**2))),
            "tolerance": atol,
            "weights_source": str(POLICY),
            "teacher_updates": 0,
        }
