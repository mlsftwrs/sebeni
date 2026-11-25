"""Tests for corpus normalization, DabaX phantom-parse guard, strict_parsed_rate, curation, and taq multi-clitics."""

from __future__ import annotations

from pathlib import Path

import pytest

from beni.core import Analysis, Sentence, Token
from beni.core.compute.metrics import MorphologyScorer
from beni.core.morphotactic.dabax import DabaX, clear_dabax_cache, get_dabax
from beni.core.srl.config import DataConfig
from beni.data.datasets import SebeniDataLoader
from beni.utils.curation import curation_summary, export_gloss_pending
from beni.utils.normalize import clean_corpus_lines, is_clean_corpus_line, normalize_homoglyphs


def test_normalize_homoglyphs_and_ascii_noop():
    # Greek/Cyrillic homoglyphs -> African Latin
    assert normalize_homoglyphs("sɛnu sԑnu sεnu sↄↄ sͻͻ") == "sɛnu sɛnu sɛnu sɔɔ sɔɔ"
    assert normalize_homoglyphs("n-ӑddәwlӑt Ɣisa γisa") == "n-ăddəwlăt Ɣisa ɣisa"
    # Clean Latin/ASCII/apostrophes unchanged
    clean = "A y'a fɔ k’a bɛ taa ɲɛ."
    assert normalize_homoglyphs(clean) == clean
    assert normalize_homoglyphs("") == ""
    assert normalize_homoglyphs(None) == ""


def test_is_clean_corpus_line_filters_noise():
    assert is_clean_corpus_line("Hakilina wɛrɛw bɛ yen aw tangacogo kan.") is True
    assert is_clean_corpus_line("laxaheeri Noun. sorghum, variety. Category: 6.2.") is False
    assert is_clean_corpus_line("après le sujet túŋ précède le prédicatif") is False
    assert is_clean_corpus_line("[Seq enter.Pfv [house inside]]") is False
    assert clean_corpus_lines(["sɛnu sԑnu", "Active participle ends in i"]) == ["sɛnu sɛnu"]


def test_sebeni_data_loader_normalization_and_noise_filter(tmp_path: Path):
    p = tmp_path / "bam.txt"
    p.write_text(
        "sεnu sԑnu\nlaxaheeri Noun. sorghum.\nActive participle in the sentence.\n",
        encoding="utf-8",
    )
    # Default: homoglyphs normalized, noise lines preserved unless opt-in filter enabled
    loader_default = SebeniDataLoader(DataConfig())
    recs_default = loader_default.load(p)
    assert len(recs_default) == 3
    assert recs_default[0]["text"] == "sɛnu sɛnu"

    loader_filtered = SebeniDataLoader(DataConfig(filter_corpus_noise=True))
    recs_filtered = loader_filtered.load(p)
    assert recs_filtered == [{"text": "sɛnu sɛnu", "lang": "bam"}]


def test_dabax_phantom_parse_guard_and_strict_flag(tmp_path: Path):
    pytest.importorskip("daba.mparser")
    clear_dabax_cache()
    # 'zzqzzq' splits at stage 0 or 1 in dtm (-i / -a / etc. or no) — construct minimal gram/dict
    gram = tmp_path / "g.gram"
    ldict = tmp_path / "d.dict"
    gram.write_text(
        "plan\nfor token:\nstage 0 add parallel parse infl\nstage 0 apply lookup\nreturn if parsed\n"
        "stage 1 apply firstmatch parse foreign\nreturn if parsed\n\n"
        "section infl\npattern :n: [ {|nu}:: ] | :n: [ :n: :mrph:PL ]\n\n"
        "section foreign\npattern <re>.*x.*</re>:: | ::EMPR\n",
        encoding="utf-8",
    )
    ldict.write_text(
        "\\_sh v3.0 400 MDF 4.0\n\n\\lang dtm\n\\name test\n\\ver 0.0.1\n\n"
        "\\lx muso\n\\ps n\n\\ge woman\n",
        encoding="utf-8",
    )

    dx_strict = get_dabax("dtm", gram=gram, ldict=ldict, runtime_dir=tmp_path / "rt1", strict_parse=True)
    # Real stem + suffix -> stage 0
    assert str(dx_strict.loader("musonu")[0].tokens[0].stage) == "0"
    # Unglossed phantom split ('foobarnu' where 'foobar' is not in dict) -> demoted to -1
    assert str(dx_strict.loader("foobarnu")[0].tokens[0].stage) == "-1"
    # EMPR still preserved at stage 1
    assert str(dx_strict.loader("wax")[0].tokens[0].stage) == "1"

    # Legacy strict_parse=False preserves daba.newmorph raw stage >= 0
    dx_raw = get_dabax("dtm", gram=gram, ldict=ldict, runtime_dir=tmp_path / "rt2", strict_parse=False)
    assert str(dx_raw.loader("foobarnu")[0].tokens[0].stage) == "0"


def test_strict_parsed_rate_metric():
    scorer = MorphologyScorer()
    sent = Sentence(
        text="muso wax foobarnu 12",
        lang="bam",
        tokens=[
            Token(surface="muso", stage=0, analyses=[Analysis(form="muso", ps=["n"], gloss="woman")]),
            Token(surface="wax", stage=6, analyses=[Analysis(form="wax", ps=[], gloss="EMPR")]),
            Token(surface="foobarnu", stage=0, analyses=[Analysis(form="foobarnu", ps=["n"], gloss="")]),
            Token(surface="12", stage="tokenizer", analyses=[Analysis(form="12", ps=["num"], gloss="CARDINAL")]),
        ],
    )
    assert scorer.strict_parsed_rate(sent) == pytest.approx(0.5)


def test_curation_summary_and_export_gloss_pending():
    summary = curation_summary()
    assert len(summary) == 13
    assert summary["taq"]["total"] > summary["taq"]["gloss_pending"] > 0
    top = export_gloss_pending("taq", top_n=5)
    assert len(top) == 5
    assert top[0]["freq"] >= top[-1]["freq"]
    assert all(item["lx"] and item["ps"] for item in top)


def test_taq_multiclitic_and_circumfix_parses(tmp_path: Path):
    pytest.importorskip("daba.mparser")
    dx = DabaX("taq", runtime_dir=tmp_path / "taq")
    text = "făl-a-s ḍarăt-a-wen wӑr-t-illa jer-iman-năwăn n-kăl-ălyăhud i-Măssinăɣ ĂLXAQQĂN D-ILUZAM"
    sents = dx.loader(text)
    toks = {t.surface: t for s in sents for t in s.tokens}
    # 'wӑr-t-illa' (Cyrillic ӑ) is normalized by DabaX to 'wăr-t-illa' (Latin ă)
    for word in ["făl-a-s", "ḍarăt-a-wen", "wăr-t-illa", "jer-iman-năwăn", "n-kăl-ălyăhud", "i-Măssinăɣ", "ĂLXAQQĂN", "D-ILUZAM"]:
        assert str(toks[word].stage) not in ("-1", ""), f"Expected valid parse for {word}, got {toks[word]}"
