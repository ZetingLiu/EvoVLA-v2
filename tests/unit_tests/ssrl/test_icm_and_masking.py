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
"""Regression tests added during code review (2026-08-02).

1. ICM must run with the DEFAULT config dims (action_mlp_dim=128 !=
   hidden_dim=512): the forward-dynamics input is latent + action embedding,
   not latent + hidden.  The original wiring crashed on every forward.
2. ``normalize_masked`` must keep boundary entries at exactly zero: a plain
   ``(x - mean) / std`` shifts the deliberately-zeroed entries to
   ``-mean/std`` (large for the nonnegative curiosity MSE).
"""

import torch

from rlinf.ssrl.icm import ICM
from rlinf.ssrl.intrinsic import (
    RunningMeanStd,
    boundary_mask_from_dones,
    normalize_masked,
)


def test_icm_forward_default_dims():
    """Default yaml dims (35 / 128 / 512) must produce a valid forward."""
    icm = ICM(latent_dim=512, action_dim=35, action_mlp_dim=128, hidden_dim=512)
    z = torch.randn(4, 512)
    a = torch.randn(4, 35)
    out = icm(z, a)
    assert out.shape == (4, 512)
    loss = icm.forward_loss(z, torch.randn(4, 512), a)
    assert torch.isfinite(loss)


def test_icm_forward_nondefault_dims():
    icm = ICM(latent_dim=64, action_dim=10, action_mlp_dim=32, hidden_dim=96)
    out = icm(torch.randn(3, 64), torch.randn(3, 10))
    assert out.shape == (3, 64)


def test_normalize_masked_keeps_invalid_zero():
    """Masked entries must stay exactly zero after mean_std normalization.

    Uses a strongly positive-mean branch (like the curiosity MSE): a plain
    normalize would map the zeroed boundary entries to ``-mean/std`` != 0.
    """
    torch.manual_seed(0)
    T, B = 10, 4
    x = torch.rand(T, B) * 5.0 + 3.0  # mean >> 0
    invalid = torch.zeros(T, B, dtype=torch.bool)
    invalid[0] = True
    invalid[7, 2] = True
    x = x.masked_fill(invalid, 0.0)

    rms = RunningMeanStd()
    out = normalize_masked(x, invalid, rms, mode="mean_std", clip=0.0)

    assert (out[invalid] == 0).all()
    # sanity: a plain normalize WOULD have shifted those zeros away from 0
    plain = rms.normalize(x, mode="mean_std")
    assert (plain[invalid].abs() > 0.1).all()
    # valid entries are actually normalized (roughly centered)
    assert abs(out[~invalid].mean().item()) < 0.5


def test_normalize_masked_scalar_stats():
    """Statistics must be scalar (flattened valid entries), so a different
    batch layout on the next update cannot raise a shape mismatch."""
    rms = RunningMeanStd()
    x1 = torch.rand(8, 4)
    x2 = torch.rand(6, 10)  # different [T, B] layout
    normalize_masked(x1, torch.zeros(8, 4, dtype=torch.bool), rms)
    normalize_masked(x2, torch.zeros(6, 10, dtype=torch.bool), rms)
    assert rms.mean.shape == ()


def test_boundary_mask_from_dones():
    T, B, C = 5, 2, 3
    dones = torch.zeros(T + 1, B, C, dtype=torch.bool)
    dones[3] = True  # transition into frame 3 is terminal
    mask = boundary_mask_from_dones(dones, T)
    assert mask.shape == (T, B)
    assert mask[0].all()  # first frame has no predecessor
    assert mask[3].all()
    assert not mask[1].any() and not mask[2].any() and not mask[4].any()
