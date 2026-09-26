"""Language identity for Sebeni (ISO / group code / optional glottocode)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

from beni.utils import config as cfg


# Mali's 13 national languages, as Sebeni group codes.
MULTI13: List[str] = [
    "bam",
    "bmq",
    "boz",
    "dtm",
    "ful",
    "mey",
    "kao",
    "myk",
    "mku",
    "spp",
    "ses",
    "snk",
    "taq",
]
OUTLIER_CODE = "bbo"
SCOPE_PRESETS = {"multi13", "all"}
# Applied only when reading the experiment jsonl. Global ``mlq`` stays Maninka.
EXPERIMENT_CODE_MAP = {"mlq": "kao", "hsy": "mey", "seq": "spp"}


def parse_lang_codes(value: Optional[Union[str, Iterable[str]]]) -> List[str]:
    """Split CLI/YAML language values (``bam,mku`` or ``["bam", "mku"]``) into codes."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
        return [p.strip().lower() for p in parts if p.strip()]
    out: List[str] = []
    for item in value:
        out.extend(parse_lang_codes(str(item)))
    return out


@dataclass
class ScopeResolution:
    """Language scope for one run.

    ``label`` is ``MULTI13``, ``SINGLE_LANG``, ``OUTLIER``, or ``PARTIAL``.
    ``dropped`` lists codes removed so they are not mixed into MULTI13 (``bbo``).
    """

    label: str
    languages: List[str]
    dropped: List[str]


def resolve_scope(value: Optional[Union[str, Iterable[str]]]) -> ScopeResolution:
    """Expand ``multi13`` / ``all`` and separate the ``bbo`` outlier.

    One ``--lang`` value may be ``multi13``, ``all``, a single code, or a
    comma-separated list. Repeating the flag still works.
    """
    tokens = parse_lang_codes(value)
    if not tokens:
        return ScopeResolution("SINGLE_LANG", ["bam"], [])
    if any(token in SCOPE_PRESETS for token in tokens):
        dropped = ["bbo"] if "bbo" in tokens else []
        return ScopeResolution("MULTI13", list(MULTI13), dropped)

    ordered: List[str] = []
    seen = set()
    for token in tokens:
        if token == OUTLIER_CODE:
            group = OUTLIER_CODE
        else:
            group = Language.from_code(token).group_code
        if group not in seen:
            seen.add(group)
            ordered.append(group)

    dropped: List[str] = []
    if OUTLIER_CODE in ordered and len(ordered) > 1:
        dropped.append(OUTLIER_CODE)
        ordered = [code for code in ordered if code != OUTLIER_CODE]

    if ordered == [OUTLIER_CODE]:
        label = "OUTLIER"
    elif len(ordered) == 1:
        label = "SINGLE_LANG"
    elif ordered == list(MULTI13) or set(ordered) == set(MULTI13):
        label = "MULTI13"
        ordered = list(MULTI13)
    else:
        label = "PARTIAL"
    return ScopeResolution(label, ordered, dropped)


def experiment_group_code(code: str) -> str:
    """Map an experiment-file language code onto a Sebeni group code.

    ``mlq`` in ``dataset_300_samples.jsonl`` is Kassonke (``kao``). That remap
    does not apply to ``Language.from_code``, which still treats ``mlq`` as
    Maninka (``mku``).
    """
    raw = str(code or "").strip().lower()
    mapped = EXPERIMENT_CODE_MAP.get(raw, raw)
    if mapped == OUTLIER_CODE:
        return OUTLIER_CODE
    return Language.from_code(mapped).group_code


@dataclass
class Language:
    """ISO-639 and Sebeni group identity for an extremely low-resource language.

    Parameters
    ----------
    iso : str
        ISO 639-3 (or 639-1) code supplied by the user or dataset.
    group_code : str
        Sebeni group code used for Distiller/DabaX baselines (e.g. ``mku`` not ``mlq``).
    glottocode : str, optional
        Glottolog identifier when known.
    name : str, optional
        Display name (language or group).
    """

    iso: str
    group_code: str
    glottocode: Optional[str] = None
    name: Optional[str] = None

    def __str__(self) -> str:
        return self.group_code or self.iso

    @classmethod
    def from_code(cls, code: str) -> "Language":
        """Build a Language from an ISO or Sebeni group/variant code.

        Maninka metadata uses group_code ``mku`` while packaged baselines live
        under ``baselines/mlq/``; both codes resolve to the same group.
        """
        needle = str(code or "").strip().lower() or "bam"
        meta = cfg.get_language_metadata(needle)
        if meta:
            group = str(meta.get("group_code") or needle).lower()
            iso = needle if len(needle) in (2, 3) else group
            return cls(
                iso=iso,
                group_code=group,
                glottocode=meta.get("glottocode"),
                name=meta.get("language") or meta.get("group_name") or group,
            )
        iso_row = cfg.get_language_iso(needle)
        group = cfg.get_group_code(needle) or needle
        name = iso_row.get("name") if iso_row else needle
        return cls(iso=needle, group_code=group, name=name)

    @classmethod
    def group_codes(cls, codes: Optional[Union[str, Iterable[str]]]) -> List[str]:
        """Unique Sebeni group codes, preserving order."""
        seen = set()
        ordered: List[str] = []
        for code in parse_lang_codes(codes):
            group = cls.from_code(code).group_code
            if group not in seen:
                seen.add(group)
                ordered.append(group)
        return ordered
