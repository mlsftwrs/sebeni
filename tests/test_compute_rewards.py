"""Issue 2: RewardManager.compute_rewards (JAX path) must not crash."""

import json
from unittest.mock import MagicMock, patch

from beni.core.compute.rewards import RewardManager
from beni.core.srl.config import RewardConfig


def _completion(text: str) -> str:
    return json.dumps(
        {
            "text": text,
            "lang": "bam",
            "tokens": [
                {
                    "surface": text.split()[0],
                    "stage": 0,
                    "analyses": [{"form": text.split()[0], "ps": ["n"], "gloss": "", "morphemes": []}],
                }
            ],
        }
    )


def test_compute_rewards_returns_all_keys():
    rm = RewardManager(reward_config=RewardConfig())
    completions = [_completion("muso ka taa"), _completion("ce bɛ na")]
    reference = [json.loads(c) for c in completions]
    dabax = MagicMock()
    dabax.loader.return_value = []
    with patch.object(RewardManager, "_dabax", return_value=dabax):
        out = rm.compute_rewards(
            completions,
            prompts=[{"role": "user", "content": "x"}] * 2,
            languages=["bam", "bam"],
            reference=reference,
        )
    for key in ("format", "morph", "rule", "lang", "total"):
        assert len(out[key]) == 2, key
    assert all(score > 0 for score in out["format"])
    assert all(score > 0 for score in out["rule"])
