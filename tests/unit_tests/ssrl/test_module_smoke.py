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
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

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
            "observation/image": torch.randint(
                0, 255, (T, B, 16, 16, 3), dtype=torch.uint8
            ),
            "observation/wrist_image": torch.randint(
                0, 255, (T, B, 8, 8, 3), dtype=torch.uint8
            ),
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


def test_pose_mode_r_cur_skipped_without_scene_obs(ssrl_module, caplog):
    """Rollout side ran without the SSRL scene_obs hook -> r_cur is None,
    the rest of the intrinsic pipeline (r_con) still works — but a warning
    must fire (review P1-6: silent degrade reads as "SSRL ran fine")."""
    batch = _fake_batch()
    del batch["forward_inputs"]["scene_obs"]
    with caplog.at_level("WARNING", logger="rlinf.ssrl"):
        intrinsic, metrics = ssrl_module.compute_intrinsic_rewards(batch)
        # fixed key set (P0-4): the key exists with 0.0, the 0/1 flag tells it
        # apart from a genuinely computed zero reward
        assert metrics["ssrl/r_cur_norm"] == 0.0
        assert metrics["ssrl/r_cur_active"] == 0.0
        assert metrics["ssrl/r_con_active"] == 1.0
        assert torch.isfinite(intrinsic).all()
        assert "r_cur is enabled but its inputs are missing" in caplog.text
        # one-time warning: a second call must not spam again
        caplog.clear()
        ssrl_module.compute_intrinsic_rewards(batch)
        assert "r_cur is enabled but its inputs are missing" not in caplog.text


def test_r_cur_nonnegative_with_default_scale_only(ssrl_module):
    """P0-2 regression: default normalize_r_cur=scale_only keeps the
    forward-MSE branch nonnegative (mean_std made it a per-step penalty in
    the 2026-08-18 smoke)."""
    batch = _fake_batch()
    T, B, C = batch["rewards"].shape
    r_cur = ssrl_module._compute_r_cur(batch["forward_inputs"], batch["dones"], T, B)
    assert r_cur is not None
    assert (r_cur.sum(dim=-1) >= 0).all(), "scale_only r_cur must stay >= 0"


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
        not torch.equal(b, a) for b, a in zip(before_icm, ssrl_module.icm.parameters())
    )
    assert icm_changed, "pose-space ICM did not receive a pretext update"


def test_ssrl_state_roundtrip(ssrl_module, monkeypatch):
    batch = _fake_batch()
    ssrl_module.compute_intrinsic_rewards(batch)
    state = ssrl_module.ssrl_state_dict()
    assert "rms_state" in state

    fresh = _make_module(monkeypatch, icm_space="pose")
    fresh.load_ssrl_state(state)
    assert fresh.rms_r_con.count == ssrl_module.rms_r_con.count
    assert fresh.rms_state.count == ssrl_module.rms_state.count

    # checkpoints written before the pose-space change lack rms_state
    legacy = {k: v for k, v in state.items() if k != "rms_state"}
    fresh2 = _make_module(monkeypatch, icm_space="pose")
    fresh2.load_ssrl_state(legacy)
    assert fresh2.rms_state.count == 0


def test_nn_module_state_dict_is_not_shadowed(ssrl_module):
    """P1-8 regression: the SSRL payload must NOT hijack ``nn.Module``'s
    ``state_dict`` / ``load_state_dict`` — generic utilities (FSDP wrapping,
    EMA helpers) call those with ``prefix``/``strict`` kwargs and expect a
    flat tensor dict."""
    flat = ssrl_module.state_dict()
    assert all(isinstance(v, torch.Tensor) for v in flat.values())
    assert any(k.startswith("visual.") for k in flat)
    # keyword contract of the base class is intact
    ssrl_module.load_state_dict(flat, strict=True)


class _RowRecorder(TorchDispatchMode):
    """Record the largest frame-stack ever materialized inside the block.

    Peak SSRL memory is proportional to the number of image rows touched in
    one pretext step; unlike a device assertion this is observable on a
    CPU-only test run (``.to("cpu")`` of a CPU tensor is a no-op).
    """

    def __init__(self):
        self.max_rows = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor) and t.dim() == 4 and t.dtype == torch.uint8:
                self.max_rows = max(self.max_rows, t.shape[0])
        return out


def test_pretext_never_materializes_full_frame_stack(ssrl_module):
    """P0-1 regression: the pretext step may only touch ``mini_batch_size``
    frames.  The previous implementation flattened the whole [T, B, H, W, 3]
    stack (T*B = 6144 frames at the CALVIN default, ~7.4 GB/view) before
    slicing, which OOMed on the real config."""
    batch = _fake_batch()
    T, B, _ = batch["rewards"].shape
    mini_batch_size = 8  # matches the fixture
    assert T * B > mini_batch_size, "fixture must be larger than one minibatch"

    recorder = _RowRecorder()
    with recorder:
        metrics = ssrl_module.update_pretext(batch)

    assert metrics["ssrl/pretext_active"] == 1.0
    assert metrics["ssrl/pretext_loss"] > 0.0
    assert 0 < recorder.max_rows <= mini_batch_size, (
        f"pretext materialized {recorder.max_rows} frames, "
        f"expected at most {mini_batch_size}"
    )
    assert batch["forward_inputs"]["observation/image"].device.type == "cpu"


def test_loss_mask_padding_zeroes_intrinsic(ssrl_module):
    """P0-3 regression: frames whose loss_mask is all-False (padding tail)
    get exactly zero intrinsic — and the pair departing from a padded frame
    is zeroed too (its input z[t-1] is garbage)."""
    batch = _fake_batch()
    T, B, C = batch["rewards"].shape
    loss_mask = torch.ones(T, B, C)
    loss_mask[4, 2] = 0  # one padded frame in the middle
    batch["loss_mask"] = loss_mask
    intrinsic, _ = ssrl_module.compute_intrinsic_rewards(batch)
    chunk_sums = intrinsic.sum(dim=-1)
    assert chunk_sums[4, 2] == 0
    assert chunk_sums[5, 2] == 0  # pair (4,2)->(5,2) uses padded input
    # rows not touching the padding may be nonzero (mask is surgical)
    assert torch.isfinite(intrinsic).all()


_FIXED_COMPUTE_KEYS = {
    "ssrl/r_con_active",
    "ssrl/r_con_norm",
    "ssrl/r_con_raw_mean",
    "ssrl/r_con_sign_flip_rate",
    "ssrl/r_cur_active",
    "ssrl/r_cur_norm",
    "ssrl/intrinsic_sum",
    "ssrl/intrinsic_abs_mean",
    "ssrl/r_ext",
    "ssrl/r_ext_abs_mean",
    "ssrl/rho_effective",
    "ssrl/time_compute_s",
}


def test_fixed_metric_keys_when_branch_disabled(monkeypatch):
    """P0-4 regression: the metric key set must not depend on which
    branches produced data (cross-rank all_reduce_dict packs sorted(keys))."""
    from omegaconf import OmegaConf

    from rlinf.ssrl.encoder import R3MVisualEncoder
    from rlinf.ssrl.module import SSRLModule

    monkeypatch.setattr(
        R3MVisualEncoder, "_load_r3m_backend", staticmethod(lambda: _StubR3M())
    )
    cfg = OmegaConf.create(
        {
            "enable": True,
            "rho": 0.6,
            "num_action_chunks": 5,
            "use_contrastive": False,  # r_con branch off
            "use_curiosity": True,
            "freeze_backbone": True,
            "normalize_intrinsic": True,
            "clip_intrinsic": 1.0,
            "encode_micro_batch": 16,
            "encoder": {"latent_dim": 512, "proj_dim": 512},
            "pretext": {"updates_per_iteration": 1, "mini_batch_size": 8},
            "icm": {"space": "pose", "state_dim": 31},
        }
    )
    module = SSRLModule(cfg, torch.device("cpu"))
    _, metrics = module.compute_intrinsic_rewards(_fake_batch())
    assert _FIXED_COMPUTE_KEYS <= set(metrics), (
        f"missing keys: {_FIXED_COMPUTE_KEYS - set(metrics)}"
    )
    assert metrics["ssrl/r_con_active"] == 0.0
    assert metrics["ssrl/r_con_norm"] == 0.0


def test_update_pretext_fixed_keys_and_warning_on_missing(ssrl_module, caplog):
    """P0-4/P1-6: a skipped pretext step still returns the fixed key set
    and warns (once) when the inputs are missing."""
    batch = _fake_batch()
    del batch["forward_inputs"]["observation/image"]
    with caplog.at_level("WARNING", logger="rlinf.ssrl"):
        metrics = ssrl_module.update_pretext(batch)
    assert set(metrics) == {
        "ssrl/pretext_loss",
        "ssrl/time_pretext_s",
        "ssrl/pretext_active",
    }
    assert metrics["ssrl/pretext_active"] == 0.0
    assert metrics["ssrl/pretext_loss"] == 0.0
    assert "pretext is enabled but its inputs are missing" in caplog.text


def test_rho_effective_is_nan_when_extrinsic_is_all_zero(ssrl_module):
    """P2-13: an all-zero r_ext batch (every task failed) used to divide by
    the 1e-6 epsilon and report a ratio in the millions, which reads exactly
    like reward hacking.  The ratio is undefined -> NaN, with the two
    absolute magnitudes carrying the signal instead."""
    batch = _fake_batch()  # rewards are all zero
    _, metrics = ssrl_module.compute_intrinsic_rewards(batch)
    assert metrics["ssrl/rho_effective"] != metrics["ssrl/rho_effective"]  # NaN
    assert metrics["ssrl/r_ext_abs_mean"] == 0.0
    assert metrics["ssrl/intrinsic_abs_mean"] > 0.0

    batch["rewards"] = torch.ones_like(batch["rewards"])
    _, metrics = ssrl_module.compute_intrinsic_rewards(batch)
    assert metrics["ssrl/rho_effective"] == pytest.approx(
        metrics["ssrl/intrinsic_abs_mean"], rel=1e-5
    )


def test_intrinsic_keeps_reward_dtype(ssrl_module):
    """The intrinsic tensor is accumulated in fp32 but must come back in the
    reward dtype, so ``rewards + rho * intrinsic`` cannot silently promote
    the reward tensor (review P2-12)."""
    batch = _fake_batch()
    batch["rewards"] = batch["rewards"].to(torch.bfloat16)
    intrinsic, _ = ssrl_module.compute_intrinsic_rewards(batch)
    assert intrinsic.dtype == torch.bfloat16


def test_freeze_encoder_trains_only_the_icm(monkeypatch):
    """P1-5: ``freeze_encoder`` was read by nothing at all — the real-robot
    config silently kept training the projection head."""
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
            "freeze_encoder": True,
            "encode_micro_batch": 16,
            "encoder": {"latent_dim": 512, "proj_dim": 512, "positive_window": 2},
            "pretext": {"updates_per_iteration": 2, "mini_batch_size": 8},
            "icm": {"space": "pose", "state_dim": 31},
        }
    )
    module = SSRLModule(cfg, torch.device("cpu"))
    assert not any(p.requires_grad for p in module.visual.parameters())
    visual_ids = {id(p) for p in module.visual.parameters()}
    optimized = {
        id(p)
        for group in module.pretext_optimizer.param_groups
        for p in group["params"]
    }
    assert not (optimized & visual_ids), "frozen visual params must not be optimized"

    batch = _fake_batch()
    before_proj = [p.clone() for p in module.visual.projection.parameters()]
    before_icm = [p.clone() for p in module.icm.parameters()]
    module.update_pretext(batch)
    assert all(
        torch.equal(b, a)
        for b, a in zip(before_proj, module.visual.projection.parameters())
    ), "freeze_encoder must keep the projection head fixed"
    assert any(
        not torch.equal(b, a) for b, a in zip(before_icm, module.icm.parameters())
    ), "the ICM must still train under freeze_encoder"


def test_pretext_loss_backpropagates_infonce_and_icm(ssrl_module):
    """Direct coverage of the production InfoNCE/lang-align/ICM objective.

    ``intrinsic.info_nce_loss`` is a reference helper that ``_pretext_loss``
    does not call, so the helper's unit test proved nothing about the loss
    that actually trains (review P2-11)."""
    torch.manual_seed(0)
    n = 8
    images = torch.randint(0, 255, (n, 16, 16, 3), dtype=torch.uint8)
    wrist = torch.randint(0, 255, (n, 8, 8, 3), dtype=torch.uint8)
    lang = F.normalize(torch.randn(n, 512), dim=-1)
    action = torch.randn(n, 35)
    traj = torch.zeros(n, dtype=torch.long)  # single trajectory: no seams
    states = torch.randn(n, 31)

    loss = ssrl_module._pretext_loss(
        images,
        lang,
        wrist,
        action,
        traj,
        temperature=0.1,
        positive_window=2,
        lang_align_coef=0.5,
        icm_states=states,
    )
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in ssrl_module.visual.projection.parameters()
    ), "InfoNCE/lang-align must reach the projection head"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in ssrl_module.icm.parameters()
    ), "the ICM forward loss must reach the ICM"

    # One trajectory per row: no InfoNCE positive and no ICM (t -> t+1) pair
    # survives the seam mask, so with lang_align off the objective is exactly
    # zero — the seam guard is what keeps rewards from being computed across
    # two different episodes.
    all_seams = torch.arange(n)
    seam_loss = ssrl_module._pretext_loss(
        images,
        lang,
        wrist,
        action,
        all_seams,
        temperature=0.1,
        positive_window=2,
        lang_align_coef=0.0,
        icm_states=states,
    )
    assert seam_loss.item() == 0.0
