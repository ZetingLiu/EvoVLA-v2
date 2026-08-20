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
"""Guard rails for the SSRL config contract (``_validate_ssrl_cfg``).

Every failure mode here degrades SSRL into a *silent* baseline: the run
completes, the ``ssrl/*`` metrics look plausible, and the ablation is
worthless.  The validator is the only thing standing between a typo and a
wasted multi-hour ABC->D run, so it needs coverage of its own — and of the
shipped yaml, which must still pass once both enables are flipped on.
"""

import pathlib

import pytest
from omegaconf import OmegaConf

from rlinf.config import _validate_ssrl_cfg

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_EMBODIED_PATH = _REPO_ROOT / "examples" / "embodiment"


def _cfg(**overrides):
    """Minimal (cfg, model_cfg) pair that passes validation unchanged."""
    base = {
        "algorithm": {
            "reward_type": "chunk_level",
            "ssrl": {"enable": True, "rho": 0.6, "num_action_chunks": 5},
        }
    }
    cfg = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))
    model_cfg = OmegaConf.create(
        {"num_action_chunks": 5, "openpi": {"ssrl": {"enable": True}}}
    )
    return cfg, model_cfg


def test_consistent_config_passes():
    cfg, model_cfg = _cfg()
    _validate_ssrl_cfg(cfg, model_cfg, only_eval=False)


def test_disabled_ssrl_skips_all_checks():
    """enable=false must be byte-for-byte the original RLinf behavior, even
    with an otherwise contradictory ssrl block."""
    cfg, model_cfg = _cfg(
        algorithm={
            "reward_type": "step_level",
            "ssrl": {"enable": False, "rho": -1.0, "num_action_chunks": 99},
        }
    )
    _validate_ssrl_cfg(cfg, model_cfg, only_eval=False)


def test_eval_only_skips_checks():
    cfg, model_cfg = _cfg(algorithm={"reward_type": "step_level"})
    _validate_ssrl_cfg(cfg, model_cfg, only_eval=True)


def test_desynced_rollout_enable_raises():
    """The actor reads ``algorithm.ssrl.enable``, the rollout-side openpi hook
    reads ``actor.model.openpi.ssrl.enable``.  Only the second one writes
    ``lang_emb`` / ``scene_obs``, so a desync means no intrinsic inputs."""
    cfg, model_cfg = _cfg()
    model_cfg.openpi.ssrl.enable = False
    with pytest.raises(ValueError, match="openpi.ssrl.enable"):
        _validate_ssrl_cfg(cfg, model_cfg, only_eval=False)


def test_missing_openpi_block_raises():
    cfg, _ = _cfg()
    with pytest.raises(ValueError, match="openpi.ssrl.enable"):
        _validate_ssrl_cfg(cfg, OmegaConf.create({"num_action_chunks": 5}), False)


def test_non_chunk_reward_type_raises():
    """Intrinsic is filled into [T, B, C] before the chunk-dim sum."""
    cfg, model_cfg = _cfg(algorithm={"reward_type": "step_level"})
    with pytest.raises(ValueError, match="chunk_level"):
        _validate_ssrl_cfg(cfg, model_cfg, only_eval=False)


def test_chunk_count_mismatch_raises():
    cfg, model_cfg = _cfg(algorithm={"ssrl": {"num_action_chunks": 4}})
    with pytest.raises(ValueError, match="num_action_chunks"):
        _validate_ssrl_cfg(cfg, model_cfg, only_eval=False)


def test_negative_rho_raises():
    cfg, model_cfg = _cfg(algorithm={"ssrl": {"rho": -0.1}})
    with pytest.raises(ValueError, match="rho"):
        _validate_ssrl_cfg(cfg, model_cfg, only_eval=False)


def test_shipped_yaml_passes_when_ssrl_is_switched_on(monkeypatch):
    """The shipped config ships with ``enable: false``; flipping both enables
    (the documented way to turn SSRL on) must satisfy the validator."""
    monkeypatch.setenv("EMBODIED_PATH", str(_EMBODIED_PATH))
    import hydra

    with hydra.initialize_config_dir(
        config_dir=str(_EMBODIED_PATH / "config"), version_base="1.1"
    ):
        cfg = hydra.compose(config_name="calvin_abc_d_ssrl_openpi_pi05")

    cfg.algorithm.ssrl.enable = True
    cfg.actor.model.openpi.ssrl.enable = True
    _validate_ssrl_cfg(cfg, cfg.actor.model, only_eval=False)
