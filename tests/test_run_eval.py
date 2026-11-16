"""run_eval scoring: empty references (issue 18)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from beni.core import Sentence, Token
from beni.core.srl.config import MasterConfig


def _sentence(text: str) -> Sentence:
    if text.startswith("EMPTY"):
        return Sentence(text=text, lang="bam", tokens=[])
    return Sentence(
        text=text,
        lang="bam",
        tokens=[Token(surface=word, stage=0) for word in text.split()],
    )


@pytest.fixture
def eval_env(tmp_path):
    """MasterConfig in a temp workdir, with DabaX mocked to split on spaces."""
    mc = MasterConfig()
    mc.working_dir = str(tmp_path)
    mc.apply_workdir(cli_path=tmp_path)
    mc.data.languages = ["bam"]
    dabax = MagicMock()
    dabax.loader.side_effect = lambda text: [_sentence(text)]
    with patch("beni.core.morphotactic.dabax.get_dabax", return_value=dabax), patch(
        "beni.core.morphotactic.dabax.resolve_baseline_files", return_value=(None, None, "ck")
    ):
        yield mc


def _rows(*texts):
    return [{"text": t, "lang": "bam"} for t in texts]


def test_empty_reference_is_skipped_and_counted(eval_env):
    from beni.core.pipeline import run_eval

    eval_env.experiment.max_eval_rows = 10
    report = run_eval(
        eval_env,
        records=_rows("a b c", "EMPTY row"),
        generate=lambda text, _lang: "a b x",
    )
    row = report["by_language"]["bam"]
    assert row["n_scored"] == 1
    assert row["n_skipped"] == 1
    assert report["n_skipped"] == 1
    assert row["n_morphemes"] == 3
    assert math.isclose(report["mer"], 1 / 3)


def test_empty_reference_reward_is_finite():
    from beni.core.compute.rewards import RewardManager
    from beni.core.srl.config import RewardConfig

    rm = RewardManager(reward_config=RewardConfig())
    completion = '{"text": "a", "lang": "bam", "tokens": [{"surface": "a", "stage": 0, "analyses": []}]}'
    reference = [{"text": "", "tokens": []}]
    scores = rm.reward_morph([completion], language=["bam"], reference=reference)
    assert scores == [0.0]


@pytest.mark.parametrize("max_rows, expected", [(None, 3), (0, 3), (1, 1), (2, 2)])
def test_max_eval_rows_caps_decoding(eval_env, max_rows, expected):
    """Issue 8: null / 0 scores every held-out row; an int caps the rows per language."""
    from beni.core.pipeline import run_eval

    eval_env.experiment.max_eval_rows = max_rows
    calls = []

    def _decode(text, _lang):
        calls.append(text)
        return text

    report = run_eval(eval_env, records=_rows("a b", "c d", "e f"), generate=_decode)
    assert len(calls) == expected
    assert report["by_language"]["bam"]["n_scored"] == expected
    assert report["mer"] == 0.0
