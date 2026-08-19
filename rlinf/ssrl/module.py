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

import time
from typing import Any

import torch
import torch.nn as nn

from rlinf.ssrl.encoder import R3MVisualEncoder
from rlinf.ssrl.icm import ICM
from rlinf.ssrl.intrinsic import (
    RunningMeanStd,
    boundary_mask_from_dones,
    clip_intrinsic,
    fill_r_con_into_chunk_rewards,
    normalize_masked,
)


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
        self.use_contrastive = bool(cfg.get("use_contrastive", True))
        self.use_curiosity = bool(cfg.get("use_curiosity", True))
        self.normalize_intrinsic = bool(cfg.get("normalize_intrinsic", True))
        self.normalize_r_con = cfg.get("normalize_r_con", "mean_std")
        self.clip_value = float(cfg.get("clip_intrinsic", 1.0))
        self.freeze_backbone = bool(cfg.get("freeze_backbone", True))

        # ---- visual / language towers (lazy heavy deps; plan §6) ----
        enc_cfg = cfg.get("encoder", {})
        self.visual = R3MVisualEncoder(
            backbone=enc_cfg.get("backbone", "r3m_resnet18"),
            latent_dim=int(enc_cfg.get("latent_dim", 2048)),
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
            else int(enc_cfg.get("latent_dim", 2048))
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

        # ---- pretext optimizer (independent of the actor optimizer) ----
        pretext_lr = float(enc_cfg.get("lr", 1.0e-4))
        self.pretext_optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=pretext_lr
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

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[N, H, W, 3]`` uint8 images in micro-batches.

        The raw frames stay on CPU (the rollout batch device) and are moved
        to the SSRL GPU one slice at a time, bounding peak memory.
        """
        outs = []
        for i in range(0, images.shape[0], self.encode_micro_batch):
            outs.append(self.visual(images[i : i + self.encode_micro_batch].to(self.device)))
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

        r_con = self._compute_r_con(forward_inputs, dones, T, B)
        r_cur = self._compute_r_cur(forward_inputs, dones, T, B)

        # Intrinsic branches are computed on the SSRL device; the result is
        # moved back to the batch device so the caller can add it to the
        # (typically CPU) rollout rewards.
        intrinsic = torch.zeros(T, B, C, device=self.device, dtype=rewards.dtype)
        metrics: dict[str, Any] = {}
        if r_con is not None:
            intrinsic = intrinsic + r_con
            metrics["ssrl/r_con_norm"] = r_con.mean().item()
            metrics["ssrl/r_con_raw_mean"] = self._last_r_con_raw
            metrics["ssrl/r_con_sign_flip_rate"] = self._sign_flip_rate(r_con)
        if r_cur is not None:
            intrinsic = intrinsic + r_cur
            metrics["ssrl/r_cur_norm"] = r_cur.mean().item()
        metrics["ssrl/intrinsic_sum"] = intrinsic.sum().item()
        # §8.4 hacking watch: mean |rho * intrinsic| relative to mean |r_ext|
        # (mean-based ratio — the per-entry ratio is undefined on a sparse
        # r_ext that is mostly zero).
        metrics["ssrl/r_ext"] = rewards.mean().item()
        metrics["ssrl/rho_effective"] = float(
            (self.rho * intrinsic.abs().mean())
            / (rewards.abs().mean() + 1e-6)
        )
        metrics["ssrl/time_compute_s"] = time.perf_counter() - t_start
        return intrinsic.to(rewards.device), metrics

    def _compute_r_con(
        self,
        forward_inputs: dict,
        dones: torch.Tensor | None,
        T: int,
        B: int,
    ) -> torch.Tensor | None:
        if not self.use_contrastive:
            return None
        # openpi forward_inputs use FLAT slash keys ("observation/image"),
        # not a nested "observation" dict.
        images = forward_inputs.get("observation/image")  # [T?, B, H, W, 3] uint8
        lang_emb = forward_inputs.get("lang_emb")  # [T?, B, 512] fp32 L2
        if images is None or lang_emb is None:
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

        delta_s = torch.zeros(T, B, device=self.device)
        delta_s[1:T_im] = s[1:] - s[:-1]
        delta_s = delta_s.masked_fill(invalid, 0.0)

        self._last_r_con_raw = float(delta_s.abs().mean().item())
        if self.normalize_intrinsic:
            delta_s = normalize_masked(
                delta_s,
                invalid,
                self.rms_r_con,
                mode=self.normalize_r_con,
                clip=self.clip_value,
            )
        elif self.clip_value > 0:
            delta_s = clip_intrinsic(delta_s, self.clip_value)
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

    def _gather_pose_state(
        self, forward_inputs: dict, T: int
    ) -> torch.Tensor | None:
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
    ) -> torch.Tensor | None:
        if not self.use_curiosity:
            return None
        action = forward_inputs.get("action")  # [T?, B, 35]
        if action is None:
            return None
        if self.icm_space == "pose":
            x = self._gather_pose_state(forward_inputs, T)  # [T_x, B, D]
            if x is None:
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
            pred = self.icm(z[:-1].reshape(-1, z.shape[-1]), a[:-1].reshape(-1, a.shape[-1]))
        mse = (pred - z[1:].reshape(-1, z.shape[-1])) ** 2
        mse = mse.mean(dim=-1).reshape(T_w - 1, B)  # [T_w-1, B]

        invalid = torch.zeros(T, B, dtype=torch.bool, device=self.device)
        invalid[0] = True
        if T_w < T:
            invalid[T_w:] = True
        if dones is not None:
            invalid |= boundary_mask_from_dones(dones.to(self.device), T)

        r_cur = torch.zeros(T, B, device=self.device)
        r_cur[1:T_w] = mse
        r_cur = r_cur.masked_fill(invalid, 0.0)
        if self.normalize_intrinsic:
            r_cur = normalize_masked(
                r_cur, invalid, self.rms_r_cur, clip=self.clip_value
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
        InfoNCE + lang-align + ICM (one minibatch per step)."""
        self._pretext_iter += 1
        if (
            self.pretext_update_every_n_iters > 1
            and self._pretext_iter % self.pretext_update_every_n_iters != 0
        ):
            return {}
        t_start = time.perf_counter()
        forward_inputs = rollout_batch.get("forward_inputs", {})
        # openpi forward_inputs use FLAT slash keys ("observation/image",
        # "observation/wrist_image"), not a nested "observation" dict.
        images = forward_inputs.get("observation/image")
        wrist = forward_inputs.get("observation/wrist_image")
        lang_emb = forward_inputs.get("lang_emb")
        action = forward_inputs.get("action")
        if images is None or lang_emb is None:
            return {}

        T = min(images.shape[0], rollout_batch["rewards"].shape[0])
        n = T * images.shape[1]
        # Flatten b-major ([T, B, ...] -> [B, T, ...] -> [B*T, ...]) so
        # consecutive rows are consecutive frames of the SAME trajectory —
        # the InfoNCE window positives and the ICM (t -> t+1) pairs must stay
        # inside one episode.  A plain [T, B] -> [T*B] row-major flatten would
        # make consecutive rows different trajectories (t-major), silently
        # destroying the temporal pairing the losses rely on.
        im = (
            images[:T]
            .permute(1, 0, *range(2, images.dim()))
            .reshape(n, *images.shape[2:])
            .to(self.device)
        )
        wr = (
            wrist[:T].permute(1, 0, *range(2, wrist.dim())).reshape(n, *wrist.shape[2:]).to(self.device)
            if wrist is not None
            else None
        )
        lang = (
            lang_emb[:T].permute(1, 0, 2).reshape(n, lang_emb.shape[-1]).to(self.device).float()
        )
        act = (
            action[:T].permute(1, 0, 2).reshape(n, -1).to(self.device).float()
            if action is not None
            else None
        )
        # Pose-space ICM trains on the same standardized pose states as
        # r_cur; the running stats are only *read* here (they were updated
        # in compute_intrinsic_rewards earlier in this iteration), keeping
        # the prediction target consistent between reward and pretext loss.
        icm_states = None
        if self.use_curiosity and self.icm_space == "pose":
            x = self._gather_pose_state(forward_inputs, T)
            if x is not None and x.shape[0] == T:
                if self.rms_state.count == 0:
                    self.rms_state.update(x.reshape(-1, x.shape[-1]))
                x = self.rms_state.normalize(x)
                icm_states = x.permute(1, 0, 2).reshape(n, -1)

        enc_cfg = self.cfg.get("encoder", {})
        info_nce_temperature = float(enc_cfg.get("info_nce_temperature", 0.1))
        positive_window = int(enc_cfg.get("positive_window", 3))
        lang_align_coef = float(enc_cfg.get("lang_align_coef", 0.5))

        metrics: dict[str, Any] = {}
        mini_batch_size = int(self.cfg.get("pretext", {}).get("mini_batch_size", 256))
        losses = []
        for _ in range(self.pretext_updates_per_iteration):
            # Sample a CONTIGUOUS time window of the b-major array: a random
            # row subset would pair arbitrary frames across trajectories and
            # the InfoNCE/ICM losses would learn no temporal structure.
            if n > mini_batch_size:
                start = int(
                    torch.randint(
                        0, n - mini_batch_size + 1, (1,), device=self.device
                    ).item()
                )
                idx = torch.arange(start, start + mini_batch_size, device=self.device)
            else:
                idx = torch.arange(n, device=self.device)
            # b-major layout: global row = b * T + t, so row // T identifies
            # the trajectory.  Passed to the loss so that InfoNCE positives
            # and ICM (t -> t+1) pairs never cross a trajectory seam inside
            # the contiguous window.
            traj = idx // T
            self.pretext_optimizer.zero_grad()
            loss = self._pretext_loss(
                im[idx], lang[idx],
                wr[idx] if wr is not None else None,
                act[idx] if act is not None else None,
                traj,
                info_nce_temperature, positive_window, lang_align_coef,
                icm_states=icm_states[idx] if icm_states is not None else None,
            )
            loss.backward()
            self.pretext_optimizer.step()
            losses.append(loss.item())
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
            valid = (ar + positive_window < n)
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

    def state_dict(self) -> dict[str, Any]:
        return {
            "visual": self.visual.state_dict(),
            "icm": self.icm.state_dict(),
            "pretext_optimizer": self.pretext_optimizer.state_dict(),
            "rms_r_con": self.rms_r_con.state_dict(),
            "rms_r_cur": self.rms_r_cur.state_dict(),
            "rms_state": self.rms_state.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.visual.load_state_dict(state["visual"])
        self.icm.load_state_dict(state["icm"])
        self.pretext_optimizer.load_state_dict(state["pretext_optimizer"])
        self.rms_r_con.load_state_dict(state["rms_r_con"])
        self.rms_r_cur.load_state_dict(state["rms_r_cur"])
        # Absent in checkpoints written before the pose-space ICM change.
        if "rms_state" in state:
            self.rms_state.load_state_dict(state["rms_state"])

    def load_warmup_checkpoint(self, path: str) -> None:
        """Load an offline warmup checkpoint (plan B: normally unused)."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state)
