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
"""Unit tests for the wandb metric whitelist in ``MetricLogger``.

``runner.logger.wandb_metric_filter`` keeps the wandb dashboard clean while
tensorboard still receives the full metric set (debugging).  These tests
pin: whitelist filtering, the no-filter passthrough (zero behavior change),
and the one-time warning when the filter matches nothing.
"""

from omegaconf import OmegaConf

from rlinf.utils.metric_logger import MetricLogger


class _FakeWandb:
    """Records log() payloads without touching the network."""

    def __init__(self, record):
        self._record = record

    def log(self, data, step):
        self._record.append((dict(data), step))

    def finish(self):
        pass


class _FakeTensorboard:
    def __init__(self, record):
        self._record = record

    def log(self, data, step):
        self._record.append((dict(data), step))

    def finish(self):
        pass


def _make_logger(monkeypatch, wandb_metric_filter):
    wandb_record, tb_record = [], []
    monkeypatch.setattr(
        MetricLogger,
        "_create_logger_bundle",
        lambda self, **kwargs: {
            "wandb": _FakeWandb(wandb_record),
            "tensorboard": _FakeTensorboard(tb_record),
        },
    )
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {
                    "log_path": "/tmp/unused",
                    "logger_backends": ["wandb", "tensorboard"],
                    "wandb_metric_filter": wandb_metric_filter,
                },
                "per_worker_log": False,
            }
        }
    )
    return MetricLogger(cfg), wandb_record, tb_record


def test_wandb_filter_keeps_only_whitelisted_keys(monkeypatch):
    logger, wandb_record, tb_record = _make_logger(
        monkeypatch, ["env/success_once", "actor/policy_loss"]
    )
    data = {"env/success_once": 0.5, "actor/policy_loss": -0.1, "noise/key": 2.0}
    logger.log(data, step=0)
    assert wandb_record == [
        ({"env/success_once": 0.5, "actor/policy_loss": -0.1}, 0)
    ]
    # Tensorboard keeps the full set for debugging.
    assert tb_record == [(data, 0)]


def test_no_filter_is_passthrough(monkeypatch):
    logger, wandb_record, _ = _make_logger(monkeypatch, None)
    data = {"env/success_once": 0.5, "noise/key": 2.0}
    logger.log(data, step=0)
    assert wandb_record == [(data, 0)]


def test_empty_filter_warns_once_when_nothing_matches(monkeypatch, capsys):
    logger, wandb_record, _ = _make_logger(monkeypatch, ["wrong/key"])
    logger.log({"env/success_once": 0.5}, step=0)
    logger.log({"env/success_once": 0.6}, step=1)
    captured = capsys.readouterr().out
    assert "wandb_metric_filter dropped" in captured
    assert captured.count("wandb_metric_filter dropped") == 1
    assert wandb_record == [({}, 0), ({}, 1)]
