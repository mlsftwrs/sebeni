"""Tests for dynamic Stage 4 / Stage 6 EMPR detection and Φ scoring."""

from __future__ import annotations

import pytest

from beni.core import Analysis, Sentence, Token
from beni.core.compute.helpers import is_empr_token, recognized_word, stage_to_phi
from beni.core.compute.metrics import MorphologyScorer
from beni.core.compute.rewards import RewardManager
from beni.core.language import MULTI13
from beni.core.morphotactic.dabax import DabaX, empr_stages_for_lang, parse_empr_stages


def test_empr_stages_across_all_13_baselines():
    assert empr_stages_for_lang("bam") == frozenset({6})
    for group in MULTI13:
        if group == "bam":
            continue
        assert empr_stages_for_lang(group) == frozenset({4}), group


def test_parse_empr_stages_ignores_commented_rules():
    gram = """
    plan
    for token:
    stage 0 apply lookup
    stage 4 apply firstmatch parse loanwords
    return if parsed

    section propername
    # pattern <re>.+</re>:: | ::EMPR

    section loanwords
    pattern <re>.+</re>:: | ::EMPR
    """
    assert parse_empr_stages(gram) == frozenset({4})


def test_stage_to_phi_and_recognized_word_per_language():
    # Bambara: stage 4 is analyzed (pos_composition3), stage 6 is EMPR.
    assert stage_to_phi(4, lang="bam") == 1.0
    assert stage_to_phi(6, lang="bam") == 0.5
    assert recognized_word(4, lang="bam") is True
    assert recognized_word(6, lang="bam") is False

    # Other baselines (e.g. Dogon, Maninka, Mamara): stage 4 is EMPR.
    for lang in ("dtm", "mku", "myk", "bmq"):
        assert stage_to_phi(3, lang=lang) == 1.0
        assert stage_to_phi(4, lang=lang) == 0.5
        assert stage_to_phi(5, lang=lang) == pytest.approx(1e-8)
        assert recognized_word(3, lang=lang) is True
        assert recognized_word(4, lang=lang) is False
        assert recognized_word(-1, lang=lang) is False


def test_scorer_phi_uses_self_empr_lang_and_token_gloss():
    scorer = MorphologyScorer(empr=0.7, eps=1e-6)
    dtm_sent = Sentence(
        text="x wax",
        lang="dtm",
        tokens=[
            Token(surface="x", stage=0, analyses=[Analysis(form="x", ps=["n"])]),
            Token(surface="wax", stage=4, analyses=[Analysis(form="wax", ps=[], gloss="EMPR")]),
        ],
    )
    assert scorer.phi(dtm_sent) == pytest.approx((1.0 + 0.7) / 2.0)

    # Even without Sentence.lang, token's EMPR gloss triggers empr weight (when stage >= 0).
    tok_only = [Token(surface="wax", stage=4, analyses=[Analysis(form="wax", ps=[], gloss="EMPR")])]
    assert is_empr_token(tok_only[0]) is True
    assert scorer.phi(tok_only) == pytest.approx(0.7)

    # Unrecognized stage < 0 always scores 0.0 even if gloss says EMPR.
    unknown_empr = [Token(surface="???", stage=-1, analyses=[Analysis(form="???", ps=[], gloss="EMPR")])]
    assert scorer.phi(unknown_empr) == 0.0


def test_reward_rule_respects_empr_stage_for_language():
    rm = RewardManager()
    rm.scorer = MorphologyScorer(empr=0.5)
    ref_dtm = {
        "text": "wax",
        "lang": "dtm",
        "tokens": [{"surface": "wax", "stage": 4, "analyses": [{"form": "wax", "ps": ["n"], "gloss": "EMPR"}]}],
    }
    comp = '{"text": "wax", "lang": "dtm", "tokens": [{"surface": "wax", "stage": 4, "analyses": [{"form": "wax", "ps": ["n"], "gloss": "EMPR"}]}]}'
    score = rm.reward_rule(prompts=[{}], completions=[comp], language=["dtm"], reference=[ref_dtm])[0]
    # consistent = 0.5 (EMPR), lemma_overlap = 1.0 -> rule_weight * 0.5 * (0.5 + 1.0)
    assert score == pytest.approx(rm.config.rule_weight * 0.75)


def test_real_dabax_empr_scores_consistently_across_bam_and_dtm():
    pytest.importorskip("daba.mparser")
    scorer = MorphologyScorer()
    for lang, expected_stage in (("bam", "6"), ("dtm", "4")):
        dx = DabaX(lang)
        assert int(expected_stage) in dx.empr_stages
        sents = dx.loader("wax")
        assert sents and str(sents[0].tokens[0].stage) == expected_stage
        assert scorer.phi(sents[0]) == pytest.approx(0.5)
