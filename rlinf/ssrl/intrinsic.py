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
"""Intrinsic reward composition: filling, normalization, clipping, mixing.

All functions here are pure (torch-only, CPU-compatible) so the temporal
alignment contract can be unit-tested without CALVIN or a GPU — see
``tests/unit_tests/ssrl/test_delta_align.py`` and ``test_intrinsic_mix.py``.
"""

from __future__ import annotations

from typing import Literal

import torch


def fill_r_con_into_chunk_rewards(
    delta_s: torch.Tensor, num_chunks: int
) -> torch.Tensor:
    """Fill a per-chunk-frame ``[T, B]`` delta into a ``[T, B, C]`` reward tensor.

    Alignment rule (locked in the SSRL plan §3.2): the chunk dimension is
    summed *before* GAE (``chunk_level`` reward type), so any layout whose
    chunk-dimension sum equals ``delta_s`` is equivalent to adding directly
    to ``[T, B, 1]``.  We put the whole delta in the **last** chunk slot and
    zero elsewhere.

    Args:
        delta_s: ``[T, B]`` per-chunk-frame delta of the language-anchored
            similarity (or curiosity reward).
        num_chunks: ``C``, the action-chunk size (e.g. 5 for CALVIN).

    Returns:
        ``[T, B, C]`` tensor such that ``out.sum(dim=-1) == delta_s``.
    """
    if delta_s.dim() != 2:
        raise ValueError(f"delta_s must be [T, B], got {tuple(delta_s.shape)}")
    out = torch.zeros(
        *delta_s.shape, num_chunks, dtype=delta_s.dtype, device=delta_s.device
    )
    out[..., -1] = delta_s
    return out


def boundary_mask_from_dones(dones: torch.Tensor, num_frames: int) -> torch.Tensor:
    """``[T, B]`` bool mask of deltas that would cross an episode boundary.

    ``delta[t, b]`` pairs frame ``t-1`` with frame ``t`` and is only defined
    inside one episode.  In the embodied pipeline ``dones`` has shape
    ``[T+1, B, C]`` where row 0 is a bootstrap zeros row and ``dones[t, b]``
    marks the transition *into* frame ``t`` as terminal (the GAE bootstraps
    step ``t-1`` -> ``t`` with ``dones[t]``).  Masked entries: the first
    frame (no predecessor) and every frame whose incoming transition is
    terminal (its delta would pair the last frame of the finished episode
    with the first frame of the next one).

    Args:
        dones: ``[T+1, B, C]`` done flags (bootstrap row at index 0).
        num_frames: ``T``, the number of chunk frames of the delta tensor.

    Returns:
        ``[T, B]`` bool tensor, ``True`` where the delta must be zeroed.
    """
    if dones.shape[0] < num_frames + 1:
        raise ValueError(
            f"dones first dim must be >= T+1 ({num_frames + 1}), "
            f"got {dones.shape[0]}"
        )
    mask = dones[:num_frames].any(dim=-1).clone()  # [T, B]: transition into t
    if num_frames > 0:
        mask[0] = True
    return mask


def zero_delta_s_at_dones(delta_s: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """Zero the deltas that would cross an episode boundary (see
    :func:`boundary_mask_from_dones` for the ``dones`` convention).

    Args:
        delta_s: ``[T, B]`` raw deltas (already ``s_t - s_{t-1}``).
        dones: ``[T+1, B, C]`` done flags (bootstrap row at index 0).

    Returns:
        Copy of ``delta_s`` with boundary-crossing entries zeroed.
    """
    if delta_s.shape[0] == 0:
        return delta_s.clone()
    mask = boundary_mask_from_dones(dones, delta_s.shape[0])
    return delta_s.masked_fill(mask, 0.0)


def normalize_masked(
    x: torch.Tensor,
    invalid: torch.Tensor,
    rms: "RunningMeanStd",
    mode: Literal["mean_std", "scale_only"] = "mean_std",
    clip: float = 0.0,
) -> torch.Tensor:
    """RMS-normalize ``x`` while keeping masked entries at exactly zero.

    Boundary / padding entries are deliberately zeroed before normalization;
    a plain ``(x - mean) / std`` would shift those zeros to ``-mean/std``
    (systematically nonzero — e.g. for the nonnegative curiosity MSE the
    mean is large, so every episode-start frame would receive a spurious
    negative reward).  This helper therefore:

    1. updates the running statistics on **valid entries only** (flattened
       to 1-D so the statistics are a scalar per reward branch, independent
       of the batch layout — plan §3.1),
    2. normalizes and clips the full tensor,
    3. re-zeroes the invalid entries.

    Args:
        x: ``[T, B]`` reward branch with invalid entries already zero.
        invalid: ``[T, B]`` bool mask, ``True`` where the entry is undefined.
        rms: the branch's :class:`RunningMeanStd` (scalar statistics).
        mode: ``mean_std`` or ``scale_only`` (plan §8.3 #1).
        clip: if > 0, clamp to ``[-clip, clip]`` after normalization.

    Returns:
        Normalized tensor with invalid entries exactly zero.
    """
    valid = ~invalid
    if valid.any():
        rms.update(x[valid].reshape(-1))
    out = rms.normalize(x, mode=mode)
    if clip > 0:
        out = clip_intrinsic(out, clip)
    return out.masked_fill(invalid, 0.0)


class RunningMeanStd:
    """Per-component running mean/std for intrinsic reward normalization.

    Each reward branch (r_con / r_cur) keeps its own instance since their
    magnitudes differ.  Uses Welford-style running statistics (updatable,
    so the state can be checkpointed and resumed without reward-scale jumps).
    """

    def __init__(self, shape=(), epsilon: float = 1e-8, device=None):
        self.shape = tuple(shape)
        self.epsilon = epsilon
        self.count = 0
        self.mean = torch.zeros(self.shape, device=device)
        self.var = torch.ones(self.shape, device=device)

    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with batch ``x`` (leading dim = batch)."""
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        # The per-column statistics are shaped by the FIRST update; a later
        # batch with a different trailing shape (e.g. resume with another
        # total_num_envs/rollout_epoch) would broadcast against stale stats
        # and silently skew the intrinsic scale — fail loudly instead.
        if self.count > 0 and self.mean.shape != batch_mean.shape:
            raise ValueError(
                f"RunningMeanStd shape mismatch: stored {tuple(self.mean.shape)} "
                f"vs batch {tuple(batch_mean.shape)} — the batch layout changed "
                "since the statistics were initialized (env count / rollout "
                "epoch changed on resume?)."
            )
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        total_count = self.count + batch_count
        if total_count == 0:
            return
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total_count)
        m2 = self.var * self.count + batch_var * batch_count
        m2 += delta**2 * (self.count * batch_count / total_count)
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(
        self, x: torch.Tensor, mode: Literal["mean_std", "scale_only"] = "mean_std"
    ) -> torch.Tensor:
        """Normalize ``x`` with the running statistics.

        ``scale_only`` skips the mean shift (keeps the potential-shaping
        character of r_con up to scaling; see plan §8.3 #1).
        """
        std = torch.sqrt(self.var + self.epsilon)
        if mode == "mean_std":
            return (x - self.mean) / std
        if mode == "scale_only":
            return x / std
        raise ValueError(f"unknown normalize mode: {mode}")

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load_state_dict(self, state: dict) -> None:
        self.count = int(state["count"])
        self.mean = state["mean"].to(torch.float32)
        self.var = state["var"].to(torch.float32)


def clip_intrinsic(x: torch.Tensor, clip: float) -> torch.Tensor:
    """Clip intrinsic rewards to ``[-clip, clip]`` (plan §3.1)."""
    return torch.clamp(x, -clip, clip)


def mix_intrinsic(
    r_ext: torch.Tensor,
    r_con: torch.Tensor | None,
    r_cur: torch.Tensor | None,
    rho: float,
) -> torch.Tensor:
    """Compose the total reward: ``r_ext + rho * (r_con + r_cur)``.

    Disabled branches contribute zero; all tensors are ``[T, B, C]`` and
    must already be normalized/clipped where configured.
    """
    intrinsic = torch.zeros_like(r_ext)
    if r_con is not None:
        intrinsic = intrinsic + r_con
    if r_cur is not None:
        intrinsic = intrinsic + r_cur
    return r_ext + rho * intrinsic
