# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for the SSRL encoder dim contract.

The real R3M resnet18 outputs **2048-d** embeddings (512 channels over a
2x2 spatial grid, flattened), NOT 512-d.  ``algorithm.ssrl.encoder.latent_dim``
feeds the ``ProjectionHead`` input dimension, so a config value of 512
silently breaks at runtime (``nn.Linear(512, 512)`` receiving 2048) — a bug
the stub-based tests in ``test_module_smoke.py`` cannot catch because their
stub outputs exactly the configured ``latent_dim``.

These tests pin the contract from both sides (pure CPU, no R3M weights):

- ``test_ssrl_yaml_latent_dim_matches_real_r3m`` — the shipped SSRL yaml
  must keep ``latent_dim: 2048`` (catches a revert to 512).
- ``test_module_compute_with_2048_stub`` / ``test_module_rejects_512_config``
  — a 2048-d stub (real-R3M-shaped) must work end-to-end, and a 512 config
  must fail loudly instead of silently mis-wiring.
"""

import os
import pathlib

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_EMBODIED_PATH = _REPO_ROOT / "examples" / "embodiment"


class _StubR3M2048(nn.Module):
    """Interface-compatible stand-in for the real R3M resnet18 (2048-d out)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2048)

    def forward(self, x, obs_shape=None):  # noqa: D102 - mirrors r3m API
        pooled = x.float().mean(dim=(2, 3)) / 255.0  # [N, 3]
        return self.lin(pooled)


def test_ssrl_yaml_latent_dim_matches_real_r3m(monkeypatch):
    """The shipped SSRL config must use the real R3M output dim (2048)."""
    monkeypatch.setenv("EMBODIED_PATH", str(_EMBODIED_PATH))
    import hydra
    from omegaconf import OmegaConf

    with hydra.initialize_config_dir(
        config_dir=str(_EMBODIED_PATH / "config"), version_base="1.1"
    ):
        cfg = hydra.compose(config_name="calvin_abc_d_ssrl_openpi_pi05")

    enc = OmegaConf.to_container(cfg.algorithm.ssrl.encoder, resolve=True)
    assert enc["latent_dim"] == 2048, (
        "encoder.latent_dim must be 2048 (real R3M resnet18 output dim); "
        "512 would crash the projection head at runtime."
    )
    assert enc["proj_dim"] == 512  # CLIP text dim, unchanged


def _fake_batch(T=4, B=2, C=5):
    torch.manual_seed(0)
    return {
        "rewards": torch.zeros(T, B, C),
        "dones": torch.zeros(T + 1, B, C, dtype=torch.bool),
        "forward_inputs": {
            "observation/image": torch.randint(0, 255, (T, B, 16, 16, 3), dtype=torch.uint8),
            "observation/wrist_image": torch.randint(0, 255, (T, B, 8, 8, 3), dtype=torch.uint8),
            "lang_emb": torch.randn(T, B, 512),
            "action": torch.randn(T, B, 35),
            "scene_obs": torch.randn(T, B, 24),
            "observation/state_ee_pos": torch.randn(T, B, 3),
            "observation/state_ee_rot": torch.randn(T, B, 3),
            "observation/state_gripper": torch.randn(T, B, 1),
        },
    }


def _make_module(monkeypatch, latent_dim: int):
    from rlinf.ssrl.encoder import R3MVisualEncoder

    monkeypatch.setattr(
        R3MVisualEncoder, "_load_r3m_backend", staticmethod(lambda: _StubR3M2048())
    )
    from rlinf.ssrl.module import SSRLModule

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "enable": True,
            "rho": 0.6,
            "num_action_chunks": 5,
            "use_contrastive": True,
            "use_curiosity": True,
            "freeze_backbone": True,
            "normalize_intrinsic": True,
            "clip_intrinsic": 1.0,
            "encode_micro_batch": 16,
            "encoder": {"latent_dim": latent_dim, "proj_dim": 512},
            "pretext": {"updates_per_iteration": 1, "mini_batch_size": 8},
            "icm": {"space": "pose", "state_dim": 31},
        }
    )
    return SSRLModule(cfg, torch.device("cpu"))


def test_module_compute_with_2048_stub(monkeypatch):
    """latent_dim=2048 + 2048-d stub: full compute path must work."""
    module = _make_module(monkeypatch, latent_dim=2048)
    intrinsic, metrics = module.compute_intrinsic_rewards(_fake_batch())
    assert intrinsic.shape == (4, 2, 5)
    assert torch.isfinite(intrinsic).all()
    assert "ssrl/r_con_norm" in metrics and "ssrl/r_cur_norm" in metrics


def test_module_rejects_512_config(monkeypatch):
    """latent_dim=512 against a real-shaped backbone must fail loudly."""
    module = _make_module(monkeypatch, latent_dim=512)
    with pytest.raises(RuntimeError, match="mat1 and mat2|input features"):
        module.compute_intrinsic_rewards(_fake_batch())
