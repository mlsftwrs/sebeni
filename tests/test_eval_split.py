"""Issue 8: the in-training hold-out follows data.eval_ratio / data.eval_rows only."""

import logging

import pytest
from datasets import Dataset

from beni.core.srl.config import MasterConfig
from beni.core.srl.plugin import AlignmentPlugin


def _plugin(**data) -> AlignmentPlugin:
    mc = MasterConfig()
    for key, value in data.items():
        setattr(mc.data, key, value)
    mc.experiment.max_eval_rows = 1  # must not influence the in-training split
    return AlignmentPlugin(mc)


def _ds(n: int = 10) -> Dataset:
    return Dataset.from_list([{"text": str(i)} for i in range(n)])


def test_eval_ratio_sets_split_size():
    train, held = _plugin(eval_ratio=0.2).split_eval(_ds(), seed=0)
    assert (len(train), len(held)) == (8, 2)


def test_eval_rows_wins_over_ratio():
    train, held = _plugin(eval_ratio=0.2, eval_rows=3).split_eval(_ds(), seed=0)
    assert (len(train), len(held)) == (7, 3)


def test_eval_rows_leaves_one_train_row():
    train, held = _plugin(eval_rows=50).split_eval(_ds(4), seed=0)
    assert (len(train), len(held)) == (1, 3)


def test_eval_rows_zero_disables_split():
    train, held = _plugin(eval_ratio=0.2, eval_rows=0).split_eval(_ds(), seed=0)
    assert held is None and len(train) == 10


def test_no_split_when_disabled():
    train, held = _plugin(eval_ratio=0.0).split_eval(_ds(), seed=0)
    assert held is None and len(train) == 10


def test_no_split_for_single_row():
    train, held = _plugin(eval_ratio=0.5).split_eval(_ds(1), seed=0)
    assert held is None and len(train) == 1


def test_eval_ratio_one_is_rejected():
    with pytest.raises(ValueError):
        _plugin(eval_ratio=1.0).split_eval(_ds(), seed=0)


def test_default_max_eval_rows_is_all():
    assert MasterConfig().experiment.max_eval_rows is None


def test_unknown_keys_warn_and_aliases_map(caplog):
    with caplog.at_level(logging.WARNING, logger="sebeni.config"):
        mc = MasterConfig.from_dict(
            {
                "preset": "mali13",
                "scope": "MALI13",
                "bogus": 1,
                "experiment": {"eval_file": "test.json"},
                "data": {"eval_rows": 5},
            }
        )
    assert mc.experiment.preset == "mali13"
    assert mc.experiment.scope == "MALI13"
    assert mc.data.eval_rows == 5
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "bogus" in messages
    assert "experiment.eval_file" in messages
    assert "preset" not in messages.replace("mali13", "")
