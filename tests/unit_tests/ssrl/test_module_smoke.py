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
"""End-to-end CPU smoke of ``SSRLModule`` with a stub visual backbone.

The real R3M dependency is only installed after user authorization (plan
§6), so these tests monkeypatch ``R3MVisualEncoder._load_r3m_backend`` with
a tiny CNN-free stub that keeps the exact interface (``forward(x,
obs_shape=...)`` on ``[N, 3, H, W]`` float [0, 255] -> ``[N, 512]``).  This
exercises the full ``compute_intrinsic_rewards`` / ``update_pretext`` data
flow on a fake ``[T, B, C]`` rollout batch — the layer the pure-function
tests cannot reach.

Curiosity runs in the default pose space (plan §3.3, 2026-08-03 decision):
the ICM forward model consumes ``scene_obs`` + ee proprio, standardized
per-dimension.  The legacy wrist-latent branch is covered by a dedicated
``icm.space=visual`` fixture.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from rlinf.ssrl.encoder import R3MVisualEncoder


class _StubR3M(nn.Module):
    """Interface-compatible stand-in for the R3M resnet."""

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.lin = nn.Linear(3, out_dim)

    def forward(self, x, obs_shape=None):  # noqa: D102 - mirrors r3m API
        pooled = x.float().mean(dim=(2, 3)) / 255.0  # [N, 3]
        return self.lin(pooled)


def _make_module(monkeypatch, icm_space: str):
    monkeypatch.setattr(
        R3MVisualEncoder, "_load_r3m_backend", staticmethod(lambda: _StubR3M())
    )
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
            "encoder": {"latent_dim": 512, "proj_dim": 512, "positive_window": 2},
            "pretext": {"updates_per_iteration": 2, "mini_batch_size": 8},
            "icm": {
                "space": icm_space,
                "state_dim": 31,
                "action_mlp_dim": 128,
                "hidden_dim": 512,
            },
        }
    )
    return SSRLModule(cfg, torch.device("cpu"))


@pytest.fixture()
def ssrl_module(monkeypatch):
    """Default configuration: pose-space curiosity (plan §3.3 B)."""
    return _make_module(monkeypatch, icm_space="pose")


@pytest.fixture()
def ssrl_module_visual(monkeypatch):
    """Legacy fallback: wrist-latent curiosity."""
    return _make_module(monkeypatch, icm_space="visual")


def _fake_batch(T=6, B=4, C=5):
    torch.manual_seed(0)
    lang = F.normalize(torch.randn(1, B, 512), dim=-1).expand(T, B, 512).contiguous()
    dones = torch.zeros(T + 1, B, C, dtype=torch.bool)
    dones[3, 1] = True  # one mid-batch episode boundary
    return {
        "rewards": torch.zeros(T, B, C),
        "dones": dones,
        "forward_inputs": {
            "observation/image": torch.randint(0, 255, (T, B, 16, 16, 3), dtype=torch.uint8),
            "observation/wrist_image": torch.randint(0, 255, (T, B, 8, 8, 3), dtype=torch.uint8),
            "lang_emb": lang,
            "action": torch.randn(T, B, 35),
            # Pose-space curiosity inputs (openpi SSRL hook + obs_processor):
            # scene_obs(24) + ee_pos(3) + ee_rot(3) + gripper(1) = 31.
            "scene_obs": torch.randn(T, B, 24),
            "observation/state_ee_pos": torch.randn(T, B, 3),
            "observation/state_ee_rot": torch.randn(T, B, 3),
            "observation/state_gripper": torch.randn(T, B, 1),
        },
    }


def test_compute_intrinsic_end_to_end(ssrl_module):
    batch = _fake_batch()
    intrinsic, metrics = ssrl_module.compute_intrinsic_rewards(batch)
    T, B, C = batch["rewards"].shape
    assert intrinsic.shape == (T, B, C)
    assert torch.isfinite(intrinsic).all()
    # boundary chunk frames contribute exactly zero (first frame + done row)
    chunk_sums = intrinsic.sum(dim=-1)
    assert (chunk_sums[0] == 0).all()
    assert chunk_sums[3, 1] == 0
    for key in (
        "ssrl/r_con_norm",
        "ssrl/r_cur_norm",
        "ssrl/r_ext",
        "ssrl/rho_effective",
        "ssrl/time_compute_s",
    ):
        assert key in metrics, f"missing §8.4 metric {key}"
    # pose mode standardizes the state vector -> per-dim stats were updated
    assert ssrl_module.rms_state.count > 0
    assert ssrl_module.rms_state.mean.shape == (31,)


def test_compute_intrinsic_visual_fallback(ssrl_module_visual):
    batch = _fake_batch()
    intrinsic, metrics = ssrl_module_visual.compute_intrinsic_rewards(batch)
    assert intrinsic.shape == batch["rewards"].shape
    assert torch.isfinite(intrinsic).all()
    assert "ssrl/r_cur_norm" in metrics
    # visual mode never touches the pose-state normalizer
    assert ssrl_module_visual.rms_state.count == 0


def test_pose_mode_r_cur_skipped_without_scene_obs(ssrl_module):
    """Rollout side ran without the SSRL scene_obs hook -> r_cur is None,
    the rest of the intrinsic pipeline (r_con) still works."""
    batch = _fake_batch()
    del batch["forward_inputs"]["scene_obs"]
    intrinsic, metrics = ssrl_module.compute_intrinsic_rewards(batch)
    assert "ssrl/r_cur_norm" not in metrics
    assert "ssrl/r_con_norm" in metrics
    assert torch.isfinite(intrinsic).all()


def test_pose_state_dim_mismatch_raises(ssrl_module):
    batch = _fake_batch()
    batch["forward_inputs"]["scene_obs"] = torch.randn(6, 4, 10)  # not 24
    with pytest.raises(RuntimeError, match="pose state dim mismatch"):
        ssrl_module.compute_intrinsic_rewards(batch)


def test_update_pretext_trains_projection_and_icm(ssrl_module):
    batch = _fake_batch()
    before_proj = [p.clone() for p in ssrl_module.visual.projection.parameters()]
    before_icm = [p.clone() for p in ssrl_module.icm.parameters()]
    metrics = ssrl_module.update_pretext(batch)
    assert "ssrl/pretext_loss" in metrics
    assert "ssrl/time_pretext_s" in metrics
    proj_changed = any(
        not torch.equal(b, a)
        for b, a in zip(before_proj, ssrl_module.visual.projection.parameters())
    )
    assert proj_changed, "projection head did not receive a pretext update"
    icm_changed = any(
        not torch.equal(b, a)
        for b, a in zip(before_icm, ssrl_module.icm.parameters())
    )
    assert icm_changed, "pose-space ICM did not receive a pretext update"


def test_state_dict_roundtrip(ssrl_module, monkeypatch):
    batch = _fake_batch()
    ssrl_module.compute_intrinsic_rewards(batch)
    state = ssrl_module.state_dict()
    assert "rms_state" in state

    fresh = _make_module(monkeypatch, icm_space="pose")
    fresh.load_state_dict(state)
    assert fresh.rms_r_con.count == ssrl_module.rms_r_con.count
    assert fresh.rms_state.count == ssrl_module.rms_state.count

    # checkpoints written before the pose-space change lack rms_state
    legacy = {k: v for k, v in state.items() if k != "rms_state"}
    fresh2 = _make_module(monkeypatch, icm_space="pose")
    fresh2.load_state_dict(legacy)
    assert fresh2.rms_state.count == 0
