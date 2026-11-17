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


def _analysis(text, tokens):
    return json.dumps({"text": text, "lang": "bam", "tokens": tokens})


def _tok(surface, stage=0, form=None):
    return {"surface": surface, "stage": stage, "analyses": [{"form": form or surface, "ps": ["n"], "gloss": "", "morphemes": []}]}


def test_rule_reward_does_not_pay_for_echoing_the_input():
    """Issue 16: copying the sentence into "text" with empty / garbage tokens scores below a correct analysis."""
    rm = RewardManager(reward_config=RewardConfig())
    reference = [json.loads(_analysis("muso ka taa", [_tok("muso"), _tok("ka", 0), _tok("taa", 1)]))]
    correct = _analysis("muso ka taa", [_tok("muso"), _tok("ka", 0), _tok("taa", 1)])
    echo_empty = _analysis("muso ka taa", [])
    echo_garbage = _analysis("muso ka taa", [_tok("x", -1), _tok("y", -1)])
    dabax = MagicMock()
    dabax.loader.side_effect = lambda text: []  # the old code re-parsed "text"; must not matter now
    with patch.object(RewardManager, "_dabax", return_value=dabax):
        good, empty, garbage = (
            rm.reward_rule(prompts=[{}], completions=[c], language=["bam"], reference=reference)[0]
            for c in (correct, echo_empty, echo_garbage)
        )
    assert good > 0
    assert empty == 0.0 and garbage == 0.0
    dabax.loader.assert_not_called()


def test_rule_reward_parses_prompt_when_no_reference():
    from beni.core import Sentence, Token

    rm = RewardManager(reward_config=RewardConfig())
    ref = Sentence(text="muso", lang="bam", tokens=[Token(surface="muso", stage=0)])
    dabax = MagicMock()
    dabax.loader.return_value = [ref]
    prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "muso"}]
    with patch.object(RewardManager, "_dabax", return_value=dabax):
        score = rm.reward_rule(prompts=[prompt], completions=[_analysis("muso", [_tok("muso")])], language=["bam"])
    dabax.loader.assert_called_once_with("muso")
    assert score[0] > 0


def test_rule_reward_does_not_pay_for_trivial_stage_zero_split():
    """Review follow-up (#16): splitting the prompt into words at stage 0 with made-up analyses earns less."""
    rm = RewardManager(reward_config=RewardConfig())
    ref_tokens = [
        {"surface": "muso", "stage": 0, "analyses": [{"form": "muso", "ps": ["n"], "gloss": "woman", "morphemes": []}]},
        {"surface": "ye", "stage": 0, "analyses": [{"form": "ye", "ps": ["pm"], "gloss": "PFV.TR", "morphemes": []}]},
        {"surface": "taa", "stage": 0, "analyses": [{"form": "taa", "ps": ["v"], "gloss": "go", "morphemes": []}]},
    ]
    reference = [{"text": "muso ye taa", "lang": "bam", "tokens": ref_tokens}]
    correct = _analysis("muso ye taa", ref_tokens)
    trivial = _analysis(
        "muso ye taa",
        [{"surface": w, "stage": 0, "analyses": [{"form": w, "ps": ["x"], "gloss": "", "morphemes": []}]} for w in ("muso", "ye", "taa")],
    )
    good = rm.reward_rule(prompts=[{}], completions=[correct], language=["bam"], reference=reference)[0]
    gamed = rm.reward_rule(prompts=[{}], completions=[trivial], language=["bam"], reference=reference)[0]
    assert gamed < 0.75 * good
