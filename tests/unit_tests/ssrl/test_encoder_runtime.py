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
"""Runtime compatibility tests for SSRL encoders."""

import torch.nn as nn


def test_clip_text_encoder_uses_eager_attention(monkeypatch):
    """CLIP must avoid the SDPA path unsupported by torch-npu 2.6."""
    import transformers

    seen = {}

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            seen["tokenizer_model"] = model_name
            return cls()

    class _FakeTextModel(nn.Module):
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            seen["model_name"] = model_name
            seen.update(kwargs)
            return cls()

    monkeypatch.setattr(transformers, "CLIPTokenizer", _FakeTokenizer)
    monkeypatch.setattr(transformers, "CLIPTextModel", _FakeTextModel)

    from rlinf.ssrl.encoder import CLIPTextEncoder

    CLIPTextEncoder("openai/clip-vit-base-patch32", device="cpu")

    assert seen["tokenizer_model"] == "openai/clip-vit-base-patch32"
    assert seen["model_name"] == "openai/clip-vit-base-patch32"
    assert seen["attn_implementation"] == "eager"
