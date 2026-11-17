"""Issue 4: the max_mer and language promote gates see real values."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from beni.core.morphotactic.distil.distillation import DistillProposal, Distiller
from beni.core.safety.governor import SafetyGovernor
from beni.core.safety.spec import SafetySpec
from beni.core.srl.algorithm1 import distill_language


def _distiller(proposal: DistillProposal):
    d = MagicMock()
    d.phi_on_texts.return_value = 0.2
    d.propose.return_value = proposal
    return d


def _proposal(**kw) -> DistillProposal:
    base = dict(gram_text="g", dict_text="d", phi=0.2, phi_prime=0.6, parseable=True, first_create=False)
    base.update(kw)
    return DistillProposal(**base)


def test_mer_above_max_is_refused():
    d = _distiller(_proposal(mer=1.5))
    decision = distill_language(["a b"], d, SafetyGovernor(SafetySpec(max_mer=1.0)), tau=0.5)
    assert decision.allowed is False and decision.reason == "mer_exceeded"
    d.write_checkpoint.assert_not_called()


def test_mer_within_bound_promotes():
    d = _distiller(_proposal(mer=0.3))
    decision = distill_language(["a b"], d, SafetyGovernor(SafetySpec(max_mer=1.0)), tau=0.5)
    assert decision.allowed is True
    d.write_checkpoint.assert_called_once()


def test_language_mismatch_is_refused():
    d = _distiller(_proposal(language_ok=False, mer=0.1))
    decision = distill_language(["a b"], d, SafetyGovernor(SafetySpec()), tau=0.5)
    assert decision.reason == "language_mismatch"


@pytest.fixture
def bam_distiller(tmp_path):
    with patch("beni.core.morphotactic.distil.distillation.create_provider", return_value=MagicMock()):
        return Distiller(lang_code="bam", backend="algorithmic", working_dir=str(tmp_path))


@pytest.mark.parametrize(
    "candidate, current, ok",
    [
        ("\\_sh v3\n\\lang bam\n\\lx a\n", "\\lang bam\n", True),
        ("\\lang dtm\n\\lx a\n", "\\lang bam\n", False),
        ("\\lx a\n", "\\lang bam\n", False),  # header dropped by the candidate
        ("\\lx a\n", "\\lx b\n", True),  # neither has a header
    ],
)
def test_dict_language_ok(bam_distiller, candidate, current, ok):
    assert bam_distiller.dict_language_ok(candidate, current) is ok


def test_candidate_mer_counts_edits(bam_distiller):
    parses = {"current": [["a", "b"], ["c"]], "candidate": [["a", "x"], ["c"]]}
    calls = iter(["current", "candidate"])
    with patch.object(Distiller, "_parse_first", side_effect=lambda texts, g, d: parses[next(calls)]):
        mer = bam_distiller._candidate_mer(["t1", "t2"], "g", "d", "G", "D")
    assert mer == pytest.approx(1 / 3)


def test_candidate_mer_real_parse_identical_is_zero(tmp_path):
    """Real DabaX parse of the packaged bam baseline: an unchanged {G, D} has MER 0."""
    from pathlib import Path

    distiller = Distiller(lang_code="bam", backend="algorithmic", working_dir=str(tmp_path))
    distiller.handle_baselines()
    gram, ldict = Path(distiller.gram_path), Path(distiller.dict_path)
    mer = distiller._candidate_mer(
        ["muso ka taa sugu la"], gram, ldict, gram.read_text(encoding="utf-8"), ldict.read_text(encoding="utf-8")
    )
    assert mer == 0.0
    assert distiller.dict_language_ok(ldict.read_text(encoding="utf-8")) is True


def test_stub_only_candidate_does_not_raise_phi(tmp_path):
    """Issue 15: \\ps x / auto: stubs alone cannot make Φ′ > Φ."""
    from beni.core import Analysis, Sentence, Token

    distiller = Distiller(lang_code="bam", backend="algorithmic", working_dir=str(tmp_path))

    def loader_for(stubbed: bool):
        def loader(text):
            tokens = [
                Token(surface="muso", stage=0, analyses=[Analysis(form="muso", ps=["n"], gloss="woman")]),
                Token(
                    surface="zzq",
                    stage=0 if stubbed else -1,
                    analyses=[Analysis(form="zzq", ps=["x"], gloss="auto:zzq")] if stubbed else [],
                ),
            ]
            return [Sentence(text=text, lang="bam", tokens=tokens)]

        return loader

    current = MagicMock(loader=loader_for(False))
    candidate = MagicMock(loader=loader_for(True))
    with patch.object(Distiller, "_load_dabax", side_effect=[current, candidate]):
        phi = distiller.phi_on_texts(["muso zzq"], gram="g", ldict="d")
        phi_prime = distiller.phi_on_texts(["muso zzq"], gram="G", ldict="D")
    assert phi_prime == phi
    decision = SafetyGovernor(SafetySpec()).allow_promote(phi, phi_prime, parseable=True, first_create=False)
    assert decision.allowed is False and decision.reason == "phi_not_improved"


def test_real_analysis_still_counts():
    from beni.core import Analysis, Sentence, Token

    token = Token(surface="muso", stage=0, analyses=[Analysis(form="muso", ps=["n"], gloss="woman"), Analysis(form="muso", ps=["x"], gloss="auto:muso")])
    Distiller.neutralize_stubs([Sentence(text="muso", lang="bam", tokens=[token])])
    assert token.stage == 0
