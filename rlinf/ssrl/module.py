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
"""SSRLModule facade: compute intrinsic rewards / update pretext / ckpt.

Deployment contract (plan §5.3):
- Lives on the actor's local GPU (``self.device``), fp32 forward, NOT in
  FSDP, NOT in the actor optimizer, NOT in weight sync.
- Each actor rank keeps its own copy and updates it independently (v1 does
  not sync pretext gradients across ranks; RMS is per-rank too — data is
  from the same source, statistics are approximately equal, accepted and
  documented).
- ``state_dict`` includes encoder/ICM weights, pretext optimizer state and
  per-branch RMS statistics so resume does not shift the reward scale.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn

from rlinf.ssrl.encoder import R3MVisualEncoder
from rlinf.ssrl.icm import ICM
from rlinf.ssrl.intrinsic import (
    RunningMeanStd,
    apply_deadband,
    boundary_mask_from_dones,
    clip_intrinsic,
    fill_r_con_into_chunk_rewards,
    linear_reward_scale,
    mix_intrinsic,
    normalize_masked,
    smooth_similarity_ema,
)

# Plain stdlib logger: ``rlinf.ssrl`` must stay importable without the
# scheduler (Worker.logger would drag it in), and WARNING always surfaces
# through the root/lastResort handler in Ray worker logs.
_LOGGER = logging.getLogger("rlinf.ssrl")


class SSRLModule(nn.Module):
    """Self-supervised intrinsic reward module for the embodied actor.

    Args:
        cfg: the ``algorithm.ssrl`` OmegaConf dict (from the actor's full
            config).  Expected keys follow plan §6.
        device: local GPU device of the actor rank.
    """

    def __init__(self, cfg: Any, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_chunks = int(cfg.get("num_action_chunks", 5))
        self.rho = float(cfg.get("rho", 0.6))
        rho_con = cfg.get("rho_con", None)
        rho_cur = cfg.get("rho_cur", None)
        self.rho_con = self.rho if rho_con is None else float(rho_con)
        self.rho_cur = self.rho if rho_cur is None else float(rho_cur)
        schedule_cfg = cfg.get("reward_schedule", {})
        self.r_con_schedule = schedule_cfg.get("r_con", {})
        self.r_cur_schedule = schedule_cfg.get("r_cur", {})
        self._reward_iter = 0
        self.use_contrastive = bool(cfg.get("use_contrastive", True))
        self.use_curiosity = bool(cfg.get("use_curiosity", True))
        self.normalize_intrinsic = bool(cfg.get("normalize_intrinsic", True))
        self.normalize_r_con = cfg.get("normalize_r_con", "mean_std")
        # r_cur is a forward-MSE and therefore nonnegative by construction:
        # `mean_std` would subtract the mean and turn average-novelty steps
        # into a per-step penalty (plan §8.3 #1, confirmed in the 2026-08-18
        # GPU smoke where r_cur_norm went negative).  Default scale_only.
        self.normalize_r_cur = cfg.get("normalize_r_cur", "scale_only")
        for name, mode in (
            ("normalize_r_con", self.normalize_r_con),
            ("normalize_r_cur", self.normalize_r_cur),
        ):
            if mode not in ("mean_std", "scale_only"):
                raise ValueError(
                    f"algorithm.ssrl.{name} must be 'mean_std' or "
                    f"'scale_only', got {mode!r}"
                )
        self.clip_value = float(cfg.get("clip_intrinsic", 1.0))
        self.s_ema_beta = float(cfg.get("s_ema_beta", 0.0))
        self.r_con_deadband = float(cfg.get("r_con_deadband", 0.0))
        self.freeze_backbone = bool(cfg.get("freeze_backbone", True))
        # Read before the first r_con write (review P1-10: the attribute was
        # previously uninitialized until _compute_r_con ran).
        self._last_r_con_raw: float = 0.0
        self._last_r_con_stats = {
            "raw_delta_mean": 0.0,
            "raw_delta_std": 0.0,
            "smoothed_delta_mean": 0.0,
            "smoothed_delta_std": 0.0,
            "positive_rate": 0.0,
            "negative_rate": 0.0,
            "deadband_rate": 0.0,
        }
        # Warn (once per branch) instead of silently degrading when a branch
        # is enabled but its inputs are missing (plan §8.3 / review P1-6):
        # a silent None reads as "SSRL ran fine" in the metrics.
        self._warned_missing: set[str] = set()

        # ---- visual / language towers (lazy heavy deps; plan §6) ----
        enc_cfg = cfg.get("encoder", {})
        self.visual = R3MVisualEncoder(
            backbone=enc_cfg.get("backbone", "r3m_resnet18"),
            latent_dim=int(enc_cfg.get("latent_dim", 512)),
            proj_dim=int(enc_cfg.get("proj_dim", 512)),
            freeze_backbone=self.freeze_backbone,
        ).to(device)

        # ---- curiosity space (plan §3.3, 2026-08-03 decision) ----
        # "pose" (default): forward dynamics on the low-dim pose state
        #   [scene_obs(24), ee_pos(3), ee_rot(3), gripper(1)].  Avoids
        #   high-dim visual prediction (the argument of the ICM paper
        #   itself) while keeping the prediction-error pretext task, and
        #   inherits EvoVLA-v1's POE intuition of exploring in pose space.
        # "visual": legacy wrist-latent branch, kept as a switchable
        #   fallback (predicts the R3M latent of the next wrist frame).
        icm_cfg = cfg.get("icm", {})
        self.icm_space = str(icm_cfg.get("space", "pose"))
        if self.icm_space not in ("pose", "visual"):
            raise ValueError(
                f"algorithm.ssrl.icm.space must be 'pose' or 'visual', "
                f"got {self.icm_space!r}"
            )
        # CALVIN: scene_obs(24) + ee_pos(3) + ee_rot(3) + gripper(1) = 31.
        self.icm_state_dim = int(icm_cfg.get("state_dim", 31))
        icm_input_dim = (
            self.icm_state_dim
            if self.icm_space == "pose"
            else int(enc_cfg.get("latent_dim", 512))
        )
        self.icm = ICM(
            latent_dim=icm_input_dim,
            # Default 35 = 5 chunks x 7 DoF (CALVIN); override with
            # algorithm.ssrl.icm.action_dim for other envs.
            action_dim=int(icm_cfg.get("action_dim", self.num_chunks * 7)),
            action_mlp_dim=int(icm_cfg.get("action_mlp_dim", 128)),
            hidden_dim=int(icm_cfg.get("hidden_dim", 512)),
            forward_loss_coef=float(icm_cfg.get("forward_loss_coef", 1.0)),
        ).to(device)

        # ---- per-branch normalizers (plan §3.1: independent RMS) ----
        # Created on the SSRL device: intrinsic tensors are computed on
        # ``self.device`` (the batch itself lives on CPU), so the running
        # statistics must live there too or update()/normalize() would
        # raise a cross-device RuntimeError.
        self.rms_r_con = RunningMeanStd(device=device)
        self.rms_r_cur = RunningMeanStd(device=device)
        # Per-DIMENSION stats for the pose state vector (trailing shape is
        # kept by RunningMeanStd): scene_obs entries (positions, joint
        # angles, binary switch states) and proprio live on very different
        # scales, so the forward-model MSE must be computed on standardized
        # coordinates or a few large-scale dims would dominate the reward.
        self.rms_state = RunningMeanStd(device=device)

        # ``freeze_encoder`` (real-robot default in the plan) freezes the
        # projection head on top of ``freeze_backbone``, leaving only the ICM
        # trainable: r_con then measures progress in a fixed metric space.
        # Previously the key was read by nothing at all (review P1-5).
        self.freeze_encoder = bool(cfg.get("freeze_encoder", False))
        if self.freeze_encoder:
            for p in self.visual.parameters():
                p.requires_grad = False
            self.visual.eval()

        # ---- pretext optimizer (independent of the actor optimizer) ----
        pretext_lr = float(enc_cfg.get("lr", 1.0e-4))
        trainable = [p for p in self.parameters() if p.requires_grad]
        # freeze_encoder + use_curiosity=false leaves nothing to train and
        # Adam would raise on an empty parameter list; degrade to "no pretext
        # update" instead of crashing mid-run.
        self.pretext_optimizer = (
            torch.optim.Adam(trainable, lr=pretext_lr) if trainable else None
        )
        if self.pretext_optimizer is None:
            _LOGGER.warning(
                "no trainable SSRL parameters (freeze_encoder=%s, "
                "use_curiosity=%s) — pretext updates are disabled and the "
                "encoders stay at their initialization.",
                self.freeze_encoder,
                self.use_curiosity,
            )
        self.pretext_updates_per_iteration = int(
            cfg.get("pretext", {}).get("updates_per_iteration", 4)
        )
        # Skip pretext updates on all but every n-th PPO iteration to reduce
        # reward non-stationarity (plan §8.3 #4).
        self.pretext_update_every_n_iters = int(
            cfg.get("pretext", {}).get("update_every_n_iters", 1)
        )
        self._pretext_iter = 0
        # Encoder micro-batch: the full [T*B] image stack (e.g. 96x64=6144
        # frames) would OOM in a single R3M forward (plan §8.3 #8).
        self.encode_micro_batch = int(cfg.get("encode_micro_batch", 256))

    def _warn_missing(self, branch: str, missing: list[str]) -> None:
        """One-time warning when an enabled branch degrades to None.

        A missing ``lang_emb`` / ``scene_obs`` / ``action`` silently
        disables the branch and its metric keys disappear — on multi-GPU
        that also desyncs the cross-rank metric key set (review P0-4).
        Warn once per (branch, missing-set) instead of spamming every step.
        """
        tag = f"{branch}:{','.join(missing)}"
        if tag in self._warned_missing:
            return
        self._warned_missing.add(tag)
        _LOGGER.warning(
            "%s is enabled but its inputs are missing (%s) — this branch is "
            "silently skipped this step. Check that the rollout-side openpi "
            "SSRL hook is enabled (actor.model.openpi.ssrl.enable) and that "
            "the env provides the keys.",
            branch,
            ", ".join(missing),
        )

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[N, H, W, 3]`` uint8 images in micro-batches.

        The raw frames stay on CPU (the rollout batch device) and are moved
        to the SSRL GPU one slice at a time, bounding peak memory.
        """
        outs = []
        for i in range(0, images.shape[0], self.encode_micro_batch):
            outs.append(
                self.visual(images[i : i + self.encode_micro_batch].to(self.device))
            )
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    # Intrinsic reward computation (no policy gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_intrinsic_rewards(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``(intrinsic [T,B,C], metrics)`` for the given rollout batch.

        Intrinsic is added to ``rewards`` *before* chunk-level GAE; this
        function never touches the policy graph (all detached).
        """
        t_start = time.perf_counter()
        rewards = rollout_batch["rewards"]  # [T, B, C]
        T, B, C = rewards.shape
        if C != self.num_chunks:
            raise RuntimeError(
                f"chunk size mismatch: batch {C} vs config {self.num_chunks} "
                "(algorithm.ssrl.num_action_chunks must match the action-chunk "
                "size used by the rollout/reward pipeline)"
            )
        dones = rollout_batch.get("dones")  # [T+1, B, C]
        forward_inputs = rollout_batch.get("forward_inputs", {})
        loss_mask = rollout_batch.get("loss_mask")  # [T, B, C] or None

        r_con = self._compute_r_con(forward_inputs, dones, T, B, loss_mask)
        r_cur = self._compute_r_cur(forward_inputs, dones, T, B, loss_mask)

        # Intrinsic branches are computed on the SSRL device; the result is
        # moved back to the batch device so the caller can add it to the
        # (typically CPU) rollout rewards.
        zeros = torch.zeros(T, B, C, device=self.device, dtype=torch.float32)
        con_scale = linear_reward_scale(self._reward_iter, self.r_con_schedule)
        cur_scale = linear_reward_scale(self._reward_iter, self.r_cur_schedule)
        rho_con_effective = self.rho_con * con_scale
        rho_cur_effective = self.rho_cur * cur_scale
        intrinsic = mix_intrinsic(
            zeros,
            r_con,
            r_cur,
            rho=self.rho,
            rho_con=rho_con_effective,
            rho_cur=rho_cur_effective,
        )
        self._reward_iter += 1
        # FIXED metric key set (review P0-4): the keys must not depend on
        # which branches produced data — cross-rank all_reduce_dict packs
        # sorted(keys) into a tensor and rank-dependent key sets would
        # desync it.  Inactive branches report 0.0 plus a 0/1 flag.
        metrics: dict[str, Any] = {}
        metrics["ssrl/r_con_active"] = float(r_con is not None)
        metrics["ssrl/r_cur_active"] = float(r_cur is not None)
        if r_con is not None:
            metrics["ssrl/r_con_norm"] = r_con.mean().item()
            metrics["ssrl/r_con_raw_mean"] = self._last_r_con_raw
            metrics["ssrl/r_con_sign_flip_rate"] = self._sign_flip_rate(r_con)
        else:
            metrics["ssrl/r_con_norm"] = 0.0
            metrics["ssrl/r_con_raw_mean"] = 0.0
            metrics["ssrl/r_con_sign_flip_rate"] = 0.0
        if r_cur is not None:
            metrics["ssrl/r_cur_norm"] = r_cur.mean().item()
        else:
            metrics["ssrl/r_cur_norm"] = 0.0
        for name, value in self._last_r_con_stats.items():
            metrics[f"ssrl/r_con_{name}"] = value if r_con is not None else 0.0
        metrics["ssrl/rho_con_effective"] = rho_con_effective
        metrics["ssrl/rho_cur_effective"] = rho_cur_effective
        metrics["ssrl/r_con_weighted_abs_mean"] = (
            (rho_con_effective * r_con).abs().mean().item()
            if r_con is not None
            else 0.0
        )
        metrics["ssrl/r_cur_weighted_abs_mean"] = (
            (rho_cur_effective * r_cur).abs().mean().item()
            if r_cur is not None
            else 0.0
        )
        metrics["ssrl/intrinsic_sum"] = intrinsic.sum().item()
        # §8.4 hacking watch: mean effective intrinsic relative to mean |r_ext|
        # (mean-based ratio — the per-entry ratio is undefined on a sparse
        # r_ext that is mostly zero).
        metrics["ssrl/r_ext"] = rewards.mean().item()
        intrinsic_abs = intrinsic.abs().mean().item()
        r_ext_abs = rewards.abs().mean().item()
        metrics["ssrl/intrinsic_abs_mean"] = intrinsic_abs
        metrics["ssrl/r_ext_abs_mean"] = r_ext_abs
        # An all-zero r_ext (every CALVIN task failed in this batch) used to
        # divide by the 1e-6 epsilon and report a ratio in the millions, which
        # reads as reward hacking (review P2-13).  The ratio is genuinely
        # undefined there: report NaN and rely on the two absolute magnitudes
        # above.  Threshold is well below any real per-step CALVIN reward.
        metrics["ssrl/rho_effective"] = (
            intrinsic_abs / r_ext_abs if r_ext_abs > 1e-8 else float("nan")
        )
        metrics["ssrl/time_compute_s"] = time.perf_counter() - t_start
        # Accumulated in fp32 (see the zeros() above) and cast back so adding
        # it to `rewards` cannot silently promote the reward tensor's dtype.
        return intrinsic.to(device=rewards.device, dtype=rewards.dtype), metrics

    def _compute_r_con(
        self,
        forward_inputs: dict,
        dones: torch.Tensor | None,
        T: int,
        B: int,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self.use_contrastive:
            return None
        # openpi forward_inputs use FLAT slash keys ("observation/image"),
        # not a nested "observation" dict.
        images = forward_inputs.get("observation/image")  # [T?, B, H, W, 3] uint8
        lang_emb = forward_inputs.get("lang_emb")  # [T?, B, 512] fp32 L2
        if images is None or lang_emb is None:
            missing = [
                name
                for name, value in (
                    ("observation/image", images),
                    ("lang_emb", lang_emb),
                )
                if value is None
            ]
            self._warn_missing("r_con", missing)
            return None
        T_im = min(images.shape[0], T)
        flat = images[:T_im].reshape(-1, *images.shape[2:])  # [T_im*B, H, W, 3]
        phi = self._encode_images(flat)  # [T_im*B, 512] L2 (micro-batched)
        proj = self.visual.projection(phi).reshape(T_im, B, -1)  # [T_im, B, 512]
        lang = lang_emb[:T_im].to(self.device).float()  # [T_im, B, 512]
        s = (proj * lang).sum(dim=-1)  # [T_im, B] cosine similarity

        # Invalid entries (kept at exactly zero, also after normalization):
        # first frame, frames beyond the image horizon, instruction switches
        # (§3.2) and episode boundaries (§3.3).
        invalid = torch.zeros(T, B, dtype=torch.bool, device=self.device)
        invalid[0] = True
        if T_im < T:
            invalid[T_im:] = True
        sim_lang = (lang[:-1] * lang[1:]).sum(dim=-1)  # [T_im-1, B]
        invalid[1:T_im] |= sim_lang < 0.999
        if dones is not None:
            invalid |= boundary_mask_from_dones(dones.to(self.device), T)
        # Padding tail (review P0-3): frames whose loss_mask is all-False
        # carry garbage observations; their nonzero intrinsic would leak into
        # earlier valid steps through the backwards GAE recursion.  A padded
        # frame poisons BOTH pairs that touch it (delta_s[t] uses s[t-1],
        # r_cur[t] uses z[t-1] as input), so mask t and t+1.
        if loss_mask is not None:
            pad = ~loss_mask[:T].any(dim=-1).to(self.device)  # [T, B]
            invalid |= pad
            if T > 1:
                invalid[1:] |= pad[:-1]

        raw_delta = torch.zeros(T, B, device=self.device)
        raw_delta[1:T_im] = s[1:] - s[:-1]
        raw_delta = raw_delta.masked_fill(invalid, 0.0)

        smoothed_s = smooth_similarity_ema(
            s,
            invalid[:T_im],
            self.s_ema_beta,
        )
        delta_s = torch.zeros(T, B, device=self.device)
        delta_s[1:T_im] = smoothed_s[1:] - smoothed_s[:-1]
        delta_s = delta_s.masked_fill(invalid, 0.0)

        self._last_r_con_raw = float(delta_s.abs().mean().item())
        valid = ~invalid
        valid_raw = raw_delta[valid]
        valid_smoothed = delta_s[valid]
        self._last_r_con_stats.update(
            {
                "raw_delta_mean": (
                    valid_raw.mean().item() if valid_raw.numel() else 0.0
                ),
                "raw_delta_std": (
                    valid_raw.std(unbiased=False).item()
                    if valid_raw.numel()
                    else 0.0
                ),
                "smoothed_delta_mean": (
                    valid_smoothed.mean().item() if valid_smoothed.numel() else 0.0
                ),
                "smoothed_delta_std": (
                    valid_smoothed.std(unbiased=False).item()
                    if valid_smoothed.numel()
                    else 0.0
                ),
            }
        )
        if self.normalize_intrinsic:
            delta_s = normalize_masked(
                delta_s,
                invalid,
                self.rms_r_con,
                mode=self.normalize_r_con,
                clip=0.0,
            )
        deadbanded = apply_deadband(delta_s, invalid, self.r_con_deadband)
        valid_before_deadband = delta_s[valid]
        deadband_count = (
            (valid_before_deadband.abs() < self.r_con_deadband).sum().item()
            if valid_before_deadband.numel()
            else 0
        )
        if self.clip_value > 0:
            delta_s = clip_intrinsic(deadbanded, self.clip_value)
        else:
            delta_s = deadbanded
        valid_final = delta_s[valid]
        valid_count = valid_final.numel()
        self._last_r_con_stats.update(
            {
                "positive_rate": (
                    (valid_final > 0).float().mean().item() if valid_count else 0.0
                ),
                "negative_rate": (
                    (valid_final < 0).float().mean().item() if valid_count else 0.0
                ),
                "deadband_rate": deadband_count / valid_count if valid_count else 0.0,
            }
        )
        return fill_r_con_into_chunk_rewards(delta_s, self.num_chunks)

    # forward_inputs keys concatenated into the pose state (plan §3.3 B).
    # scene_obs comes from the openpi SSRL hook (rollout side); the ee_*
    # keys are the CALVIN proprio slices that obs_processor already emits.
    _POSE_KEYS = (
        "scene_obs",
        "observation/state_ee_pos",
        "observation/state_ee_rot",
        "observation/state_gripper",
    )

    def _gather_pose_state(self, forward_inputs: dict, T: int) -> torch.Tensor | None:
        """Concat the pose-state parts into ``[T_x, B, state_dim]`` fp32.

        Returns None when any part is missing (e.g. the rollout side ran
        with SSRL disabled, or a non-CALVIN env without the ee_* keys).
        """
        parts = []
        for key in self._POSE_KEYS:
            value = forward_inputs.get(key)
            if value is None:
                return None
            parts.append(value.to(self.device).float())
        T_x = min(min(p.shape[0] for p in parts), T)
        x = torch.cat([p[:T_x] for p in parts], dim=-1)  # [T_x, B, D]
        if x.shape[-1] != self.icm_state_dim:
            raise RuntimeError(
                f"pose state dim mismatch: gathered {x.shape[-1]} vs configured "
                f"algorithm.ssrl.icm.state_dim={self.icm_state_dim} "
                "(set state_dim to scene_obs + proprio dims of your env)"
            )
        return x

    def _compute_r_cur(
        self,
        forward_inputs: dict,
        dones: torch.Tensor | None,
        T: int,
        B: int,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self.use_curiosity:
            return None
        action = forward_inputs.get("action")  # [T?, B, 35]
        if action is None:
            self._warn_missing("r_cur", ["action"])
            return None
        if self.icm_space == "pose":
            x = self._gather_pose_state(forward_inputs, T)  # [T_x, B, D]
            if x is None:
                missing = [
                    key for key in self._POSE_KEYS if forward_inputs.get(key) is None
                ]
                self._warn_missing("r_cur", missing or ["pose state"])
                return None
            T_w = x.shape[0]
            if T_w < 2:
                return fill_r_con_into_chunk_rewards(
                    torch.zeros(T, B, device=self.device), self.num_chunks
                )
            # Standardize per-dimension, then predict in standardized space.
            self.rms_state.update(x.reshape(-1, x.shape[-1]))
            z = self.rms_state.normalize(x)  # [T_w, B, D]
        else:
            wrist = forward_inputs.get("observation/wrist_image")  # [T?, B, H, W, 3]
            if wrist is None:
                self._warn_missing("r_cur", ["observation/wrist_image"])
                return None
            T_w = min(wrist.shape[0], T)
            if T_w < 2:
                return fill_r_con_into_chunk_rewards(
                    torch.zeros(T, B, device=self.device), self.num_chunks
                )
            z = self._encode_images(wrist[:T_w].reshape(-1, *wrist.shape[2:]))
            z = z.reshape(T_w, B, -1).detach()  # stop-grad w.r.t. policy/contrastive
        a = action[:T_w].to(self.device).float().reshape(T_w, B, -1)
        # r_cur[t] = || f(z_{t-1}, a_{t-1}) - z_t ||^2 (arrival-frame layout,
        # same convention as delta_s), only for legal in-episode pairs.
        with torch.no_grad():
            pred = self.icm(
                z[:-1].reshape(-1, z.shape[-1]), a[:-1].reshape(-1, a.shape[-1])
            )
        mse = (pred - z[1:].reshape(-1, z.shape[-1])) ** 2
        mse = mse.mean(dim=-1).reshape(T_w - 1, B)  # [T_w-1, B]

        invalid = torch.zeros(T, B, dtype=torch.bool, device=self.device)
        invalid[0] = True
        if T_w < T:
            invalid[T_w:] = True
        if dones is not None:
            invalid |= boundary_mask_from_dones(dones.to(self.device), T)
        if loss_mask is not None:
            pad = ~loss_mask[:T].any(dim=-1).to(self.device)  # [T, B]
            invalid |= pad
            if T > 1:
                invalid[1:] |= pad[:-1]

        r_cur = torch.zeros(T, B, device=self.device)
        r_cur[1:T_w] = mse
        r_cur = r_cur.masked_fill(invalid, 0.0)
        if self.normalize_intrinsic:
            r_cur = normalize_masked(
                r_cur,
                invalid,
                self.rms_r_cur,
                mode=self.normalize_r_cur,
                clip=self.clip_value,
            )
        elif self.clip_value > 0:
            r_cur = clip_intrinsic(r_cur, self.clip_value)
        return fill_r_con_into_chunk_rewards(r_cur, self.num_chunks)

    @staticmethod
    def _sign_flip_rate(x: torch.Tensor) -> float:
        """Fraction of consecutive non-zero deltas with opposite signs (§8.4)."""
        flat = x.reshape(-1)
        nz = flat[flat != 0]
        if nz.numel() < 2:
            return 0.0
        return float((nz[1:] * nz[:-1] < 0).float().mean().item())

    # ------------------------------------------------------------------
    # Pretext update (per PPO iteration, independent optimizer)
    # ------------------------------------------------------------------

    def update_pretext(self, rollout_batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Run ``pretext.updates_per_iteration`` mini-batch steps of
        InfoNCE + lang-align + ICM (one minibatch per step).

        Returns a FIXED metric key set every call (review P0-4): skipped
        steps report zeros so the cross-rank ``all_reduce_dict`` never sees
        rank-dependent key collections.
        """
        metrics: dict[str, Any] = {
            "ssrl/pretext_loss": 0.0,
            "ssrl/time_pretext_s": 0.0,
            "ssrl/pretext_active": 0.0,
        }
        self._pretext_iter += 1
        t_start = time.perf_counter()
        if self.pretext_optimizer is None:
            # Nothing trainable (freeze_encoder without curiosity).
            metrics["ssrl/time_pretext_s"] = time.perf_counter() - t_start
            return metrics
        if (
            self.pretext_update_every_n_iters > 1
            and self._pretext_iter % self.pretext_update_every_n_iters != 0
        ):
            metrics["ssrl/time_pretext_s"] = time.perf_counter() - t_start
            return metrics
        forward_inputs = rollout_batch.get("forward_inputs", {})
        # openpi forward_inputs use FLAT slash keys ("observation/image",
        # "observation/wrist_image"), not a nested "observation" dict.
        images = forward_inputs.get("observation/image")
        wrist = forward_inputs.get("observation/wrist_image")
        lang_emb = forward_inputs.get("lang_emb")
        action = forward_inputs.get("action")
        if images is None or lang_emb is None:
            missing = [
                name
                for name, value in (
                    ("observation/image", images),
                    ("lang_emb", lang_emb),
                )
                if value is None
            ]
            self._warn_missing("pretext", missing)
            metrics["ssrl/time_pretext_s"] = time.perf_counter() - t_start
            return metrics

        T = min(images.shape[0], rollout_batch["rewards"].shape[0])
        B = images.shape[1]
        n = T * B
        # b-major row index -> (b, t) mapping: consecutive rows are
        # consecutive frames of the SAME trajectory, which the InfoNCE
        # window positives and ICM (t -> t+1) pairs rely on (a t-major
        # flatten would pair different trajectories).  Instead of
        # materializing the full b-major permuted copy on the GPU (n frames
        # = T*B, ~7.4 GB per view at the default CALVIN scale — review
        # P0-1), the contiguous window is gathered on the CPU with advanced
        # indexing and only the selected mini-batch rows move to the device.
        enc_cfg = self.cfg.get("encoder", {})
        info_nce_temperature = float(enc_cfg.get("info_nce_temperature", 0.1))
        positive_window = int(enc_cfg.get("positive_window", 3))
        lang_align_coef = float(enc_cfg.get("lang_align_coef", 0.5))

        mini_batch_size = int(self.cfg.get("pretext", {}).get("mini_batch_size", 256))
        losses = []
        for _ in range(self.pretext_updates_per_iteration):
            # Contiguous b-major window (same semantics as before, but the
            # rows are gathered instead of materializing the full batch).
            if n > mini_batch_size:
                start = int(torch.randint(0, n - mini_batch_size + 1, (1,)).item())
                idx = torch.arange(start, start + mini_batch_size)
            else:
                idx = torch.arange(n)
            b_idx, t_idx = idx // T, idx % T  # b-major: row = b * T + t
            traj = b_idx.to(self.device)  # seam mask must live on proj.device
            im = images[t_idx, b_idx].to(self.device)
            lang = lang_emb[t_idx, b_idx].to(self.device).float()
            wr = wrist[t_idx, b_idx].to(self.device) if wrist is not None else None
            act = (
                action[t_idx, b_idx].to(self.device).float()
                if action is not None
                else None
            )
            # Pose-space ICM trains on the same standardized pose states as
            # r_cur; the running stats are only *read* here (they were
            # updated in compute_intrinsic_rewards earlier in this
            # iteration), keeping the prediction target consistent between
            # reward and pretext loss.
            icm_states = None
            if self.use_curiosity and self.icm_space == "pose":
                x = self._gather_pose_state(forward_inputs, T)
                if x is not None and x.shape[0] == T:
                    if self.rms_state.count == 0:
                        self.rms_state.update(x.reshape(-1, x.shape[-1]))
                    x = self.rms_state.normalize(x)
                    icm_states = x[t_idx, b_idx]
            self.pretext_optimizer.zero_grad()
            loss = self._pretext_loss(
                im,
                lang,
                wr,
                act,
                traj,
                info_nce_temperature,
                positive_window,
                lang_align_coef,
                icm_states=icm_states,
            )
            loss.backward()
            self.pretext_optimizer.step()
            losses.append(loss.item())
        metrics["ssrl/pretext_active"] = 1.0
        if losses:
            metrics["ssrl/pretext_loss"] = sum(losses) / len(losses)
        metrics["ssrl/time_pretext_s"] = time.perf_counter() - t_start
        return metrics

    def _pretext_loss(
        self,
        images: torch.Tensor,
        lang: torch.Tensor,
        wrist: torch.Tensor | None,
        action: torch.Tensor | None,
        traj: torch.Tensor | None,
        temperature: float,
        positive_window: int,
        lang_align_coef: float,
        icm_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """InfoNCE (temporal window, in-batch negatives) + language
        alignment (+ ICM forward loss if curiosity is used).

        ``traj`` identifies the trajectory of each (contiguous, b-major) row
        so that positives and ICM pairs never straddle a trajectory seam.
        ``icm_states`` are the pre-standardized pose states ([N, state_dim])
        for the pose-space ICM; when None the visual fallback encodes the
        wrist frames instead.
        """
        import torch.nn.functional as F

        # Anchor embeddings: [N, 512] L2 (projected for the con view).
        phi = self.visual(images)
        proj = self.visual.projection(phi)

        n = proj.shape[0]
        ar = torch.arange(n, device=proj.device)
        loss = torch.zeros((), device=self.device)

        # Temporal InfoNCE (plan §3.2): positive = the frame W steps ahead in
        # the SAME trajectory; negatives = the whole batch outside the
        # temporal window (frames of other trajectories included — CALVIN
        # rollouts of other envs are natural negatives).
        if self.use_contrastive and n >= positive_window + 2:
            pos_idx = (ar + positive_window).clamp(max=n - 1)
            valid = ar + positive_window < n
            if traj is not None:
                valid &= traj[pos_idx] == traj
            if valid.any():
                q = proj[valid]  # [Nv, D]
                pos = proj[pos_idx[valid]]  # [Nv, D]
                logits_all = q @ proj.t() / temperature  # [Nv, N]
                # Exclude self and the same-trajectory temporal window from
                # the negatives (they are near-positives, not negatives).
                qi = ar[valid][:, None]  # [Nv, 1]
                near = (ar[None, :] - qi).abs() <= positive_window
                if traj is not None:
                    near &= traj[None, :] == traj[valid][:, None]
                neg_logits = logits_all.masked_fill(near, float("-inf"))
                pos_logit = (q * pos).sum(dim=-1, keepdim=True) / temperature
                logits = torch.cat([pos_logit, neg_logits], dim=1)  # [Nv, 1+N]
                loss = loss - F.log_softmax(logits, dim=1)[:, 0].mean()

        # Language alignment: proj(image) ~ lang_emb (weak label, §3.2).
        if lang_align_coef > 0:
            loss = loss + lang_align_coef * (1.0 - (proj * lang).sum(dim=-1)).mean()

        # ICM forward loss — prediction target depends on icm.space:
        # pose mode uses the standardized pose states (constant target, no
        # visual forward needed); visual mode encodes the wrist frames.
        # (t -> t+1) pairs must stay inside one trajectory either way.
        if self.use_curiosity and action is not None and n >= 2:
            z = None
            if self.icm_space == "pose":
                z = icm_states  # [N, state_dim], already standardized
            elif wrist is not None:
                with torch.no_grad():
                    z = self.visual(wrist).detach()
            if z is not None:
                pair_ok = (
                    traj[1:] == traj[:-1]
                    if traj is not None
                    else torch.ones(n - 1, dtype=torch.bool, device=proj.device)
                )
                if pair_ok.any():
                    pred = self.icm(z[:-1][pair_ok], action[:-1][pair_ok])
                    loss = loss + self.icm.forward_loss_coef * nn.functional.mse_loss(
                        pred, z[1:][pair_ok]
                    )
        return loss

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    # NOTE: deliberately *not* named ``state_dict`` / ``load_state_dict``
    # (review P1-8).  SSRLModule is an nn.Module and overriding those two with
    # an incompatible payload (optimizer + RMS state, no ``prefix``/``strict``
    # kwargs) breaks any generic nn.Module utility — FSDP wrapping, EMA
    # helpers, ``torch.save(model.state_dict())`` — that walks submodules.
    def ssrl_state_dict(self) -> dict[str, Any]:
        return {
            "visual": self.visual.state_dict(),
            "icm": self.icm.state_dict(),
            "pretext_optimizer": (
                self.pretext_optimizer.state_dict()
                if self.pretext_optimizer is not None
                else None
            ),
            "rms_r_con": self.rms_r_con.state_dict(),
            "rms_r_cur": self.rms_r_cur.state_dict(),
            "rms_state": self.rms_state.state_dict(),
            "reward_iter": self._reward_iter,
        }

    def load_ssrl_state(self, state: dict[str, Any]) -> None:
        self.visual.load_state_dict(state["visual"])
        self.icm.load_state_dict(state["icm"])
        opt_state = state.get("pretext_optimizer")
        if self.pretext_optimizer is not None and opt_state is not None:
            self.pretext_optimizer.load_state_dict(opt_state)
        self.rms_r_con.load_state_dict(state["rms_r_con"])
        self.rms_r_cur.load_state_dict(state["rms_r_cur"])
        # Absent in checkpoints written before the pose-space ICM change.
        if "rms_state" in state:
            self.rms_state.load_state_dict(state["rms_state"])
        # Absent in checkpoints written before reward scheduling.
        self._reward_iter = int(state.get("reward_iter", 0))

    def load_warmup_checkpoint(self, path: str) -> None:
        """Load an offline warmup checkpoint (plan B: normally unused)."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_ssrl_state(state)
