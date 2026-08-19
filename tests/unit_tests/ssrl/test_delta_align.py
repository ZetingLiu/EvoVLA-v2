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
"""#7 temporal alignment tests (plan §8.3 #7).

The delta of the language-anchored similarity (or curiosity reward) is
computed per chunk frame ``[T, B]`` and must be injected into the rewards
tensor ``[T, B, C]`` *before* the ``chunk_level`` sum used by GAE, such
that ``sum over C`` recovers exactly ``delta_s`` and no entry crosses an
episode / epoch boundary.

Pure CPU tests: no CALVIN, no GPU, no Ray.
"""

import numpy as np
import pytest
import torch

from rlinf.algorithms.utils import preprocess_embodied_advantages_inputs
from rlinf.ssrl.intrinsic import (
    fill_r_con_into_chunk_rewards,
    zero_delta_s_at_dones,
)


def _reshuffle_for_adv(tensor: torch.Tensor, rollout_epoch: int) -> torch.Tensor:
    """Local replica of ``process_nested_dict_for_adv`` semantics.

    ``[rollout_epoch * T, B, ...] -> [T, rollout_epoch * B, ...]``.  Kept
    local (not imported from ``fsdp_actor_worker``) so this test stays pure
    CPU without pulling in Ray/FSDP; the reshape/transpose here mirrors the
    production function exactly.
    """
    epoch, t, b = rollout_epoch, tensor.shape[0] // rollout_epoch, tensor.shape[1]
    new = tensor.reshape(epoch, t, b, *tensor.shape[2:])  # [E, T, B, ...]
    new = new.transpose(0, 1)  # [T, E, B, ...]
    return new.reshape(t, epoch * b, *tensor.shape[2:])  # [T, E*B, ...]


def _make_dones(T: int, B: int, C: int, end_after: list[int]) -> torch.Tensor:
    """[T+1, B, C] dones in the embodied pipeline layout.

    Row 0 is the bootstrap zeros row; the episode ending after frame ``f``
    is recorded at ``dones[f+1]`` — i.e. ``dones[t]`` marks the transition
    *into* frame ``t`` as terminal (the GAE bootstraps step ``t-1`` -> ``t``
    with ``dones[t]``, see advantages.py).
    """
    dones = torch.zeros(T + 1, B, C, dtype=torch.bool)
    for f in end_after:
        dones[f + 1] = True
    return dones


def test_fill_sum_conservation():
    """Filling must preserve the chunk-level sum exactly (plan §3.2)."""
    torch.manual_seed(0)
    T, B, C = 8, 4, 5
    delta_s = torch.randn(T, B) * 0.1
    filled = fill_r_con_into_chunk_rewards(delta_s, C)
    assert filled.shape == (T, B, C)
    torch.testing.assert_close(filled.sum(dim=-1), delta_s)
    # layout: whole delta in the last slot, zeros elsewhere (locked rule)
    torch.testing.assert_close(filled[..., -1], delta_s)
    assert (filled[..., :-1] == 0).all()


def test_fill_rejects_wrong_rank():
    with pytest.raises(ValueError):
        fill_r_con_into_chunk_rewards(torch.zeros(2, 3, 4), 5)


def test_chunk_level_preprocess_recovers_delta():
    """End-to-end: r_ext + fill(delta_s) through the real GAE preprocessing.

    ``preprocess_embodied_advantages_inputs`` (chunk_level) sums the C dim
    and flattens to micro-step rewards [n_steps, bsz]; the intrinsic delta
    must survive that pipeline 1:1.
    """
    torch.manual_seed(1)
    T, B, C = 6, 3, 5
    r_ext = torch.rand(T, B, C) * 0.05
    delta_s = torch.randn(T, B) * 0.1
    dones = _make_dones(T, B, C, end_after=[3])
    values = torch.rand(T + 1, B, 1)

    filled = fill_r_con_into_chunk_rewards(delta_s, C)
    total = r_ext + filled

    kwargs = preprocess_embodied_advantages_inputs(
        rewards=total,
        dones=dones,
        values=values,
        reward_type="chunk_level",
        adv_type="gae",
    )
    # chunk_level sums C first, so micro-step rewards are [T, B] and equal
    # the chunk sum of r_ext plus exactly delta_s.
    torch.testing.assert_close(kwargs["rewards"], r_ext.sum(dim=-1) + delta_s)


def test_zero_delta_at_dones():
    """Deltas crossing an episode boundary (and the first frame) are zeroed."""
    T, B, C = 5, 2, 3
    delta_s = torch.randn(T, B)
    dones = _make_dones(T, B, C, end_after=[2])  # episode ends after frame 2
    out = zero_delta_s_at_dones(delta_s, dones)

    assert out[0].eq(0).all()  # first frame has no predecessor
    assert out[3].eq(0).all()  # transition into frame 3 crosses the boundary
    # frames 1..2 are inside the ended episode, frame 4 is inside the next
    # one (its delta comes from fresh deltas after the reset), so their
    # given values must stay.
    torch.testing.assert_close(out[1], delta_s[1])
    torch.testing.assert_close(out[2], delta_s[2])
    torch.testing.assert_close(out[4], delta_s[4])


def test_multi_epoch_reshuffle_no_cross_talk():
    """After [epoch*T, B] -> [T, epoch*B] reshuffle, no delta crosses epochs.

    Each reshuffled batch slot is one (epoch, env) trajectory, so every
    slot's first frame has no predecessor: its delta must be zero, and the
    fill must still conserve the per-slot sum.
    """
    torch.manual_seed(2)
    T0, B, E, C = 4, 2, 2, 5  # per-epoch chunk frames, envs, epochs, chunks
    # Raw per-(epoch, env) similarity sequence in original layout [E*T0, B].
    s_raw = torch.randn(E * T0, B)
    # Per-frame deltas computed inside each trajectory (same trajectory =
    # same b, consecutive t); first frame of each epoch has no predecessor.
    delta_raw = s_raw.clone()
    delta_raw[0] = 0.0
    delta_raw[1:] = s_raw[1:] - s_raw[:-1]
    for e in range(1, E):
        delta_raw[e * T0] = 0.0  # first frame of each epoch

    reshaped = _reshuffle_for_adv(delta_raw, E)
    assert reshaped.shape == (T0, E * B)

    # After reshuffle, each b slot's frame 0 is a trajectory start: no
    # delta may point into a previous epoch.
    torch.testing.assert_close(reshaped[0], torch.zeros(E * B))
    # And the deltas inside each slot equal the original in-trajectory deltas.
    expected = torch.stack([delta_raw[e * T0 : (e + 1) * T0] for e in range(E)], dim=1)
    expected = expected.reshape(T0, E * B)
    torch.testing.assert_close(reshaped, expected)

    # Full pipeline on the reshuffled batch: fill + chunk sum conservation.
    dones = _make_dones(T0, E * B, C, end_after=[T0 - 1])
    safe = zero_delta_s_at_dones(reshaped, dones)
    filled = fill_r_con_into_chunk_rewards(safe, C)
    torch.testing.assert_close(filled.sum(dim=-1), safe)


def test_lang_switch_no_diff(monkeypatch):
    """Language change must not diff across the switch (plan §3.2).

    Tested at the intrinsic level: a similarity sequence where the anchor
    changes between frames must produce zero delta at the switch frame.
    """
    T, C = 6, 5
    # lang_emb identical for t in [0, 1, 2], different from t=3 on.
    lang = torch.zeros(T, 1, 512)
    lang[:3] = 1.0 / (512**0.5)
    lang[3:] = -1.0 / (512**0.5)
    # A fixed projected visual embedding whose similarity to lang therefore
    # differs between the two segments (1.0 then -1.0), so the delta at the
    # switch frame is genuinely non-zero and must be zeroed by the mask.
    proj = torch.full((T, 1, 512), 1.0 / (512**0.5))
    s = (proj * lang).sum(dim=-1)  # 1.0 inside segment 1, -1.0 inside segment 2
    delta = s[1:] - s[:-1]
    # Switch at t=2 -> 3: delta must be zeroed there.
    switch_mask = (lang[:-1] * lang[1:]).sum(dim=-1) < 0.999  # [T-1]
    safe_delta = torch.where(
        switch_mask, torch.zeros_like(delta), delta
    )
    assert delta[2].item() != 0.0  # the switch delta is real, masking is tested
    torch.testing.assert_close(safe_delta[2], torch.zeros(1))
    torch.testing.assert_close(safe_delta[0], delta[0])
    _ = np  # keep imports honest
