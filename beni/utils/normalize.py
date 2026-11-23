"""Unicode homoglyph normalization and corpus noise filtering utilities."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

HOMOGLYPH_TABLE = str.maketrans(
    {
        "ε": "ɛ",
        "ԑ": "ɛ",
        "Ε": "Ɛ",
        "Ԑ": "Ɛ",
        "ͻ": "ɔ",
        "ↄ": "ɔ",
        "Ϲ": "Ɔ",
        "ӑ": "ă",
        "Ӑ": "Ă",
        "ә": "ə",
        "Ә": "Ə",
        "γ": "ɣ",
        "Γ": "Ɣ",
    }
)

_SIL_OR_META_RE = re.compile(
    r"(?:\b(?:Noun|Verb|Adj|Adv)\.\s|\b(?:Category|Main entry|See):|"
    r"\b(?:Example|Exemple|Forme pré-verbale|Construisez|3ème personne|"
    r"proposition|prédicatif|le sujet|après le|précède le|Active participle|"
    r"Nasalized|sonorants|topicalizer|Lesson \d)\b)",
    re.IGNORECASE,
)

_INTERLINEAR_RE = re.compile(r"\b(Pfv|Ipfv|Seq|Pl|Sg|3Sg|1Sg|2Sg|Def|Refl|Caus)\b")
_LATIN_WORD_RE = re.compile(r"[a-zàâçéèêëîïôûùüÿœæ]+")

_EN_FR_STOP = frozenset(
    {
        "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
        "have", "has", "been", "which", "their", "will", "would", "could", "should",
        "about", "into", "through", "when", "these", "those", "they", "them", "your",
        "not", "but", "can", "may", "all", "one", "two", "also", "used", "only",
        "verb", "noun", "adj", "adv", "pron", "prep", "postp", "conj", "interj",
        "num", "part", "sg", "pl", "pfv", "ipfv", "neg", "imp", "def", "indef", "loc",
        "she", "he", "it", "we", "you", "at", "in", "on", "by", "to", "of", "is",
        "les", "des", "une", "dans", "pour", "pas", "qui", "que", "sur", "est",
        "sont", "avec", "par", "plus", "tout", "mais", "cette", "comme", "aussi",
        "nous", "vous", "leur", "fait", "être", "avoir", "peut", "doit", "aux",
        "ses", "ces", "son", "entre", "sous", "sans", "dont", "sujet", "objet",
    }
)


def normalize_homoglyphs(text: Optional[str]) -> str:
    """Map visually identical Greek/Cyrillic homoglyphs to canonical African Latin."""
    if not text:
        return "" if text is None else text
    return text.translate(HOMOGLYPH_TABLE)


def is_clean_corpus_line(line: str) -> bool:
    """Return False when a raw line is EN/FR commentary, interlinear gloss, or SIL metadata."""
    s = (line or "").strip()
    if not s:
        return False
    if _SIL_OR_META_RE.search(s):
        return False
    if s.count("[") >= 2 or _INTERLINEAR_RE.search(s):
        return False
    words = _LATIN_WORD_RE.findall(s.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in _EN_FR_STOP)
    if hits >= 2 and hits / len(words) >= 0.22:
        return False
    return True


def clean_corpus_lines(lines: Iterable[str], *, normalize: bool = True) -> List[str]:
    """Filter noisy lines and optionally normalize Unicode homoglyphs."""
    out: List[str] = []
    for raw in lines:
        if not is_clean_corpus_line(raw):
            continue
        cleaned = raw.strip()
        if normalize:
            cleaned = normalize_homoglyphs(cleaned)
        out.append(cleaned)
    return out
