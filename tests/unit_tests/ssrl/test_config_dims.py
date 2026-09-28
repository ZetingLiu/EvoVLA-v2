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
"""Regression tests for the SSRL encoder dimension contract.

The official R3M ResNet-18 uses torchvision's global average pool and outputs
**512-d** embeddings. ``algorithm.ssrl.encoder.latent_dim`` feeds the
``ProjectionHead`` input dimension, so the config and backend must agree.

These tests pin the contract from both sides (pure CPU, no R3M weights):

- ``test_ssrl_yaml_latent_dim_matches_real_r3m`` pins the shipped YAML to 512.
- ``test_module_compute_with_512_stub`` / ``test_module_rejects_2048_config``
  verify the valid path and a clear error for a mismatched configuration.
"""

import pathlib

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_EMBODIED_PATH = _REPO_ROOT / "examples" / "embodiment"


class _StubR3M512(nn.Module):
    """Interface-compatible stand-in for the real R3M ResNet-18."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 512)

    def forward(self, x, obs_shape=None):  # noqa: D102 - mirrors r3m API
        pooled = x.float().mean(dim=(2, 3)) / 255.0  # [N, 3]
        return self.lin(pooled)


def test_ssrl_yaml_latent_dim_matches_real_r3m(monkeypatch):
    """The shipped SSRL config must use the real R3M output dimension."""
    monkeypatch.setenv("EMBODIED_PATH", str(_EMBODIED_PATH))
    import hydra
    from omegaconf import OmegaConf

    with hydra.initialize_config_dir(
        config_dir=str(_EMBODIED_PATH / "config"), version_base="1.1"
    ):
        cfg = hydra.compose(config_name="calvin_abc_d_ssrl_openpi_pi05")

    enc = OmegaConf.to_container(cfg.algorithm.ssrl.encoder, resolve=True)
    assert enc["latent_dim"] == 512, (
        "encoder.latent_dim must be 512 (real R3M ResNet-18 output dim); "
        "other values would crash the projection head at runtime."
    )
    assert enc["proj_dim"] == 512  # CLIP text dim, unchanged


def _fake_batch(T=4, B=2, C=5):
    torch.manual_seed(0)
    return {
        "rewards": torch.zeros(T, B, C),
        "dones": torch.zeros(T + 1, B, C, dtype=torch.bool),
        "forward_inputs": {
            "observation/image": torch.randint(
                0, 255, (T, B, 16, 16, 3), dtype=torch.uint8
            ),
            "observation/wrist_image": torch.randint(
                0, 255, (T, B, 8, 8, 3), dtype=torch.uint8
            ),
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
        R3MVisualEncoder, "_load_r3m_backend", staticmethod(lambda: _StubR3M512())
    )
    from omegaconf import OmegaConf

    from rlinf.ssrl.module import SSRLModule

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


def test_module_compute_with_512_stub(monkeypatch):
    """latent_dim=512 plus a 512-d stub must work end to end."""
    module = _make_module(monkeypatch, latent_dim=512)
    intrinsic, metrics = module.compute_intrinsic_rewards(_fake_batch())
    assert intrinsic.shape == (4, 2, 5)
    assert torch.isfinite(intrinsic).all()
    assert "ssrl/r_con_norm" in metrics and "ssrl/r_cur_norm" in metrics


def test_module_rejects_2048_config(monkeypatch):
    """A mismatched 2048-d configuration must fail loudly."""
    module = _make_module(monkeypatch, latent_dim=2048)
    with pytest.raises(RuntimeError, match="R3M output dimension"):
        module.compute_intrinsic_rewards(_fake_batch())
