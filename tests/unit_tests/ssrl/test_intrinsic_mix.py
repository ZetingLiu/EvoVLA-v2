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
"""RMS / clip / rho mixing tests for intrinsic rewards (plan §3.1)."""

import pytest
import torch

from rlinf.ssrl.intrinsic import (
    RunningMeanStd,
    clip_intrinsic,
    mix_intrinsic,
)


def test_running_mean_std_online():
    rms = RunningMeanStd()
    x = torch.randn(64, 4)
    rms.update(x)
    # normalized output should be ~N(0, 1) after seeing data
    norm = rms.normalize(x)
    assert torch.isfinite(norm).all()
    assert abs(norm.mean().item()) < 1e-2
    assert abs(norm.std().item() - 1.0) < 1e-1


def test_running_mean_std_incremental_equals_batch():
    rms = RunningMeanStd()
    x = torch.randn(50, 3)
    rms.update(x[:20])
    rms.update(x[20:])
    assert rms.count == 50
    # Running stats should match full-batch stats closely
    assert torch.allclose(rms.mean, x.mean(dim=0), atol=1e-6)
    assert torch.allclose(rms.var, x.var(dim=0, unbiased=False), atol=1e-6)


def test_scale_only_keeps_center():
    """scale_only must not shift the mean (potential-shaping ablation)."""
    rms = RunningMeanStd()
    x = torch.randn(32, 2) * 3.0 + 5.0
    rms.update(x)
    mean_std = rms.normalize(x, mode="mean_std")
    scale_only = rms.normalize(x, mode="scale_only")
    assert abs(scale_only.mean().item() - 5.0 / rms.var.sqrt().mean().item()) < 1.0
    assert abs(mean_std.mean().item()) < 1e-1


def test_scale_only_rejects_bad_mode():
    rms = RunningMeanStd()
    with pytest.raises(ValueError):
        rms.normalize(torch.randn(4, 2), mode="bogus")


def test_state_dict_roundtrip():
    rms = RunningMeanStd()
    rms.update(torch.randn(20, 2))
    rms2 = RunningMeanStd()
    rms2.load_state_dict(rms.state_dict())
    assert rms2.count == rms.count
    assert torch.allclose(rms2.mean, rms.mean)
    assert torch.allclose(rms2.var, rms.var)


def test_clip_intrinsic():
    x = torch.tensor([-5.0, -0.5, 0.0, 0.5, 5.0])
    out = clip_intrinsic(x, 1.0)
    assert out.min().item() == -1.0 and out.max().item() == 1.0
    torch.testing.assert_close(out[1:4], x[1:4])


def test_mix_intrinsic():
    r_ext = torch.ones(4, 2, 5)
    r_con = torch.full((4, 2, 5), 0.5)
    r_cur = torch.full((4, 2, 5), -0.25)
    mixed = mix_intrinsic(r_ext, r_con, r_cur, rho=0.6)
    expected = r_ext + 0.6 * (r_con + r_cur)
    torch.testing.assert_close(mixed, expected)


def test_mix_disabled_branches():
    r_ext = torch.ones(4, 2, 5)
    # both branches disabled -> pure external reward
    torch.testing.assert_close(mix_intrinsic(r_ext, None, None, 0.6), r_ext)
    # only r_con
    r_con = torch.full((4, 2, 5), 0.1)
    expected = r_ext + 0.6 * r_con
    torch.testing.assert_close(mix_intrinsic(r_ext, r_con, None, 0.6), expected)


def test_info_nce_loss_shapes_match_pretext():
    """Regression: ``_pretext_loss`` must pass equal-length ``q`` and
    ``negatives`` to ``info_nce_loss`` (its einsum ``nd,nmd->nm`` requires
    equal leading dims).  The construction mirrors module.py exactly:
    ``pos_idx = arange(n) + window``, ``valid = pos_idx < n``, and the
    negatives are the rolled batch indexed by ``valid`` as well.
    """
    import torch.nn.functional as F

    from rlinf.ssrl.encoder import info_nce_loss

    n, W, D = 64, 3, 512
    torch.manual_seed(0)
    proj = F.normalize(torch.randn(n, D), dim=-1)
    pos_idx = torch.arange(n) + W
    valid = pos_idx < n
    q, pos = proj[valid], proj[pos_idx[valid]]
    neg = proj.roll(W + 1, dims=0)[valid]
    assert q.shape[0] == neg.shape[0] == n - W
    loss = info_nce_loss(q, pos, neg[:, None, :], temperature=0.1)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
