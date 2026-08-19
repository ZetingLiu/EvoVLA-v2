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
"""Self-supervised RL (SSRL) intrinsic rewards.

A minimally invasive side path on top of the embodied actor: turn pretext
tasks (temporal contrastive / ICM curiosity) into dense intrinsic rewards
that are added to the sparse environment reward *before* chunk-level GAE.

Enable/disable is fully driven by ``algorithm.ssrl.enable``; when disabled
every hook in this package is a no-op and the original RLinf behavior is
unchanged.  This package must stay import-light: importing it never pulls
in Ray, CUDA or the optional CLIP/R3M dependencies (those are imported
lazily inside the encoder module).
"""

from rlinf.ssrl.module import SSRLModule

__all__ = ["SSRLModule"]
