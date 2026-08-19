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
"""SSRL visual / language encoders: R3M backbone + projection + InfoNCE.

All heavy dependencies (transformers CLIP, torchvision, the ``r3m`` pip
package) are imported **lazily** inside methods so that importing
``rlinf.ssrl`` never fails on their absence — only instantiating an encoder
does, and only when ``algorithm.ssrl.enable=true`` (i.e. after the user has
authorized installing the SSRL dependencies, plan §6).

Design (locked in the plan):
- Visual backbone: R3M (ResNet18 init), shared for static / gripper views.
- Text tower: frozen CLIP text (``openai/clip-vit-base-patch32``, 512-d),
  computed on the rollout side with a per-instruction cache.
- ``freeze_backbone`` freezes R3M (plan B, #10=B); only the projection head
  is trained online.
- InfoNCE (temporal contrastive, window ``W``) + language-alignment term;
  gradients never flow into the policy (SSRL is a separate module).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projecting backbone features into the CLIP text space."""

    def __init__(self, input_dim: int = 512, proj_dim: int = 512, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, dim=-1)


class R3MVisualEncoder(nn.Module):
    """R3M backbone (frozen by default) + trainable projection head."""

    def __init__(
        self,
        backbone: str = "r3m_resnet18",
        latent_dim: int = 2048,  # real R3M resnet18 output (512ch x 2x2 spatial)
        proj_dim: int = 512,  # CLIP text dim
        freeze_backbone: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        assert backbone == "r3m_resnet18", f"unsupported backbone: {backbone}"
        self.image_size = image_size
        self.freeze_backbone = freeze_backbone
        # Lazy: importing r3m / torchvision at module import time would
        # break the import-light contract of rlinf.ssrl.
        r3m = self._load_r3m_backend()
        # ``load_r3m`` wraps the model in ``nn.DataParallel``, which scatters
        # the input to ALL visible GPUs and gathers the output on cuda:0 —
        # the projection head lives on ``self.device`` (e.g. cuda:3), so the
        # DP wrapper would mix devices on multi-GPU actors.  Unwrap it and
        # run the plain module on ``self.device``.
        if isinstance(r3m, torch.nn.DataParallel):
            r3m = r3m.module
        self.backbone = r3m  # R3M resnet, own resize+normalize
        self.latent_dim = latent_dim
        self.projection = ProjectionHead(input_dim=latent_dim, proj_dim=proj_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()  # BatchNorm uses running stats when frozen

    @staticmethod
    def _load_r3m_backend() -> nn.Module:
        try:
            from r3m import load_r3m
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "R3M is not installed. SSRL requires `r3m` (see external/r3m) "
                "and the CLIP transformers model; install them after user "
                "authorization (plan §6)."
            ) from exc
        return load_r3m("resnet18")  # output dim 512

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``images`` ``[B, H, W, 3]`` uint8 (HWC) into ``[B, latent]``.

        The R3M package handles /255 scaling and ImageNet normalization
        internally; we convert the layout (HWC uint8 -> ``[B, 3, H, W]``
        float in ``[0, 255]``).  ``obs_shape`` is passed explicitly so the
        R3M forward takes its Resize(256) + CenterCrop(224) branch for
        non-224 inputs (e.g. CALVIN's 200x200 static / 84x84 wrist views);
        calling ``self.backbone(x)`` without it would silently use the
        normalize-only ``[3, 224, 224]`` path.
        Output is L2-normalized (pre-projection); the caller applies the
        projection head for the r_con view.
        """
        x = images.float().permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        out = self.backbone(x, obs_shape=[3, x.shape[-2], x.shape[-1]])
        out = out.reshape(out.shape[0], -1)  # [B, latent_dim]
        return F.normalize(out, dim=-1)


class CLIPTextEncoder:
    """Frozen CLIP text encoder with per-instruction caching.

    Lives on the rollout side (plan §3.2): the actor never sees raw
    instruction strings, so ``predict_action_batch`` writes the cached
    ``lang_emb`` ``[B, 512]`` fp32 L2-normalized tensor into
    ``forward_inputs``.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device="cuda"):
        try:
            import transformers  # noqa: F401  (lazy, plan §6 authorization)
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed; CLIP text encoding needs it "
                "(install after user authorization, plan §6)."
            ) from exc
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of instruction strings into ``[B, 512]`` L2-norm."""
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self.model(**tokens).pooler_output  # [B, 512]
        out = F.normalize(out.float(), dim=-1)
        return out.cpu()  # results are cached on CPU and moved by the consumer

    def encode_cached(self, texts: list[str]) -> torch.Tensor:
        """Encode with a per-instruction cache (CALVIN instruction set is small)."""
        missing = [t for t in texts if t not in self._cache]
        if missing:
            encoded = self.encode(missing)
            for text, emb in zip(missing, encoded):
                self._cache[text] = emb
        return torch.stack([self._cache[t] for t in texts])


def info_nce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE: ``-log( exp(q·p/τ) / (exp(q·p/τ) + Σ exp(q·n/τ)) )``.

    ``query``/``positive``/``negatives`` are L2-normalized ``[N, D]`` /
    ``[N, D]`` / ``[N, M, D]``.
    """
    pos_sim = (query * positive).sum(dim=-1) / temperature  # [N]
    neg_sim = torch.einsum("nd,nmd->nm", query, negatives) / temperature  # [N, M]
    logits = torch.cat([pos_sim[:, None], neg_sim], dim=-1)  # [N, 1+M]
    return -F.log_softmax(logits, dim=-1)[:, 0].mean()
