"""Integrity and DabaX regression tests for all 13 MALI13 packaged baselines."""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path

import pytest

from beni.core.language import MALI13
from beni.core.morphotactic.dabax import DabaX, empr_stages_for_lang

BASELINES_DIR = Path("beni/data/baselines")
TEST_JSON_PATH = Path("beni/data/test.json")

CANONICAL_PS = {
    "n",
    "n.prop",
    "v",
    "vq",
    "adj",
    "adv",
    "prn",
    "pers",
    "dtm",
    "pp",
    "pm",
    "prep",
    "conj",
    "part",
    "prt",
    "num",
    "intj",
    "ptcp",
    "aux",
    "cop",
    "neg",
    "mrph",
    "ideo",
    "onomat",
}


@pytest.mark.parametrize("lang", sorted(MALI13))
def test_baseline_files_exist_and_no_stale_txt_duplicates(lang: str):
    bdir = BASELINES_DIR / lang
    assert (bdir / "baseline.gram").is_file(), f"Missing {lang}/baseline.gram"
    assert (bdir / "baseline.dict").is_file(), f"Missing {lang}/baseline.dict"
    assert not (bdir / "baseline.dict.txt").exists(), f"Stale duplicate {lang}/baseline.dict.txt"
    assert not (bdir / "baseline.gram.txt").exists(), f"Stale duplicate {lang}/baseline.gram.txt"


@pytest.mark.parametrize("lang", sorted(MALI13))
def test_dictionary_mdf_hygiene_and_dictreader(lang: str):
    daba_formats = pytest.importorskip("daba.formats")
    dict_path = BASELINES_DIR / lang / "baseline.dict"
    text = dict_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Check blank-line separation before every \lx record (after the header)
    seen_first_lx = False
    for idx, line in enumerate(lines):
        if line.startswith("\\lx "):
            if seen_first_lx:
                assert idx > 0 and lines[idx - 1].strip() == "", (
                    f"{lang}/baseline.dict line {idx + 1}: missing blank line before {line!r}"
                )
            seen_first_lx = True
        elif line.startswith("\\ps "):
            ps_val = line[4:].strip()
            assert ps_val and " " not in ps_val, (
                f"{lang}/baseline.dict line {idx + 1}: invalid \\ps {ps_val!r}"
            )
            if lang != "bam":
                assert ps_val == ps_val.lower(), (
                    f"{lang}/baseline.dict line {idx + 1}: uppercase \\ps {ps_val!r}"
                )
                for part in ps_val.split("/"):
                    assert part in CANONICAL_PS, (
                        f"{lang}/baseline.dict line {idx + 1}: non-canonical \\ps {ps_val!r}"
                    )
        elif line.startswith("\\va "):
            va_val = line[4:].strip()
            assert ";" not in va_val and "[" not in va_val, (
                f"{lang}/baseline.dict line {idx + 1}: malformed \\va {va_val!r}"
            )
        elif line.startswith("\\mm "):
            mm_val = line[4:].strip()
            assert mm_val != "::", f"{lang}/baseline.dict line {idx + 1}: empty \\mm ::"
        elif line.startswith("\\ge ") or line.startswith("\\gf "):
            gl = line[4:].strip()
            assert not gl.startswith("auto:"), (
                f"{lang}/baseline.dict line {idx + 1}: forbidden auto: gloss {gl!r}"
            )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dic = daba_formats.DictReader(str(dict_path)).get()
    errors = [ln for ln in buf.getvalue().splitlines() if "Error line:" in ln]
    assert not errors, f"{lang}/baseline.dict DictReader errors: {errors[:5]}"
    assert dic.lang == lang
    assert dic.name
    assert dic.ver
    assert len(dic) >= 500


@pytest.mark.parametrize("lang", sorted(MALI13))
def test_grammar_compiles_and_test_json_coverage(lang: str, tmp_path: Path):
    daba_newmorph = pytest.importorskip("daba.newmorph")
    expected_empr = frozenset({6}) if lang == "bam" else frozenset({4})
    assert empr_stages_for_lang(lang) == expected_empr

    dx = DabaX(lang, runtime_dir=tmp_path / lang, process=False)
    assert dx.empr_stages == expected_empr

    test_json = json.loads(TEST_JSON_PATH.read_text(encoding="utf-8"))
    tkey = "mlq" if lang == "mku" else lang
    text = test_json[tkey]
    lines = [ln.strip() for ln in text.splitlines() if re.search(r"\w", ln)]

    tot = true_ok = phantom = 0
    for para in dx.mparser().parse(lines):
        for _, toks in para:
            for tok in toks:
                tt = tok.as_tuple()
                if tt[0] != "w":
                    continue
                _, stage, glosses = tt[1]
                tot += 1
                st = str(stage)
                if st == "tokenizer" or any(daba_newmorph.parsed(g) for g in glosses):
                    true_ok += 1
                elif st != "-1" and not any(getattr(g, "gloss", "") == "EMPR" for g in glosses):
                    phantom += 1

    assert tot > 0
    true_pct = 100.0 * true_ok / tot
    phantom_pct = 100.0 * phantom / tot
    assert true_pct >= 80.0, f"{lang} test.json true_parsed={true_pct:.1f}% < 80%"
    assert phantom_pct < 10.0, f"{lang} test.json phantom={phantom_pct:.1f}% >= 10%"
