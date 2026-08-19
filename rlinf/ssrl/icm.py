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
"""ICM-style curiosity (r_cur): forward dynamics prediction error.

Forward dynamics: ``z_hat_{t+1} = f(z_t, a'_t)`` where ``a'_t`` is the
flattened action chunk through a small MLP, and ``z_t`` depends on
``algorithm.ssrl.icm.space`` (plan §3.3, 2026-08-03 decision):

- ``pose`` (default): the per-dimension standardized low-dim pose state
  (CALVIN ``scene_obs`` + ee proprio, 31-d) — prediction stays out of
  pixel/latent visual space, following the classic argument that
  predicting raw perception is hard and noisy.
- ``visual`` (fallback): the R3M wrist-image embedding (stop-grad w.r.t.
  the policy).

``r_cur = ||z_hat_{t+1} - z_{t+1}||_2^2`` (forward MSE).  Pairs are only
formed inside one episode: the last frame (no ``t+1``) and any frame
after a done boundary yield ``r_cur = 0`` (plan §3.3).

The module itself is representation-agnostic: ``latent_dim`` is simply
the width of whatever ``z`` the caller feeds in.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionChunkEncoder(nn.Module):
    """MLP encoding a flattened action chunk into a compact vector."""

    def __init__(self, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class ICMForwardDynamics(nn.Module):
    """Forward dynamics ``f(z_t, a'_t) -> z_hat_{t+1}``.

    ``a'_t`` is the ActionChunkEncoder output (``action_emb_dim`` wide), so
    the first layer input is ``latent_dim + action_emb_dim`` — NOT
    ``latent_dim + hidden_dim`` (the two only coincide when the action MLP
    width equals the forward hidden width, which the defaults do not).
    """

    def __init__(self, latent_dim: int, action_emb_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, a_t], dim=-1))


class ICM(nn.Module):
    """Curiosity module: forward prediction of the state representation.

    ``z`` is either the standardized pose state (pose mode) or a wrist
    latent (visual mode); it is always ``detach``-ed before entering this
    module (stop-grad w.r.t. both the policy and the contrastive encoder),
    so the ICM head sees a fixed target representation (plan §8.3 #6).
    """

    def __init__(
        self,
        latent_dim: int = 512,
        action_dim: int = 35,
        action_mlp_dim: int = 128,
        hidden_dim: int = 512,
        forward_loss_coef: float = 1.0,
    ):
        super().__init__()
        self.action_encoder = ActionChunkEncoder(action_dim, action_mlp_dim)
        self.forward_dynamics = ICMForwardDynamics(
            latent_dim, action_emb_dim=action_mlp_dim, hidden_dim=hidden_dim
        )
        self.forward_loss_coef = forward_loss_coef

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """Predict the next latent from (detached) ``z_t`` and action ``a_t``."""
        z_t = z_t.detach()
        a = self.action_encoder(a_t)
        return self.forward_dynamics(z_t, a)

    def forward_loss(
        self, z_t: torch.Tensor, z_next: torch.Tensor, a_t: torch.Tensor
    ) -> torch.Tensor:
        """MSE between predicted and actual next latent (mean over batch)."""
        pred = self.forward(z_t, a_t)
        return nn.functional.mse_loss(pred, z_next.detach())
