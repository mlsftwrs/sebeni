"""Curation utilities for reviewing and exporting gloss-pending Tier 4 dictionary stems."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from beni.core.language import MALI13
from beni.utils import config as cfg

_GLOSS_PENDING_MARKER = "gloss-pending"
_FREQ_RE = re.compile(r"\bfreq=(\d+)\b")


def _baselines_root(baselines_dir: Optional[Union[str, Path]] = None) -> Path:
    return Path(baselines_dir) if baselines_dir else (cfg.DATA_DIR / "baselines")


def _iter_mdf_blocks(dict_path: Path) -> List[Dict[str, Any]]:
    """Parse an MDF dictionary into lightweight entry dicts without modifying files."""
    blocks: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def flush() -> None:
        nonlocal cur
        if cur and cur.get("lx"):
            blocks.append(cur)
        cur = None

    for raw_line in dict_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if not line.startswith("\\"):
            continue
        tag, _, val = line[1:].partition(" ")
        tag = tag.strip()
        val = val.strip()
        if tag in ("lx", "le"):
            flush()
            cur = {"lx": val, "ps": "", "ge": "", "gf": "", "rem": "", "xv": ""}
        elif cur is not None and tag in cur and not cur[tag]:
            cur[tag] = val
    flush()
    return blocks


def curation_summary(
    baselines_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, int]]:
    """Return total, curated, and gloss-pending ``\\lx`` counts per MALI13 language."""
    root = _baselines_root(baselines_dir)
    summary: Dict[str, Dict[str, int]] = {}
    for lang in sorted(MALI13):
        dict_path = root / lang / "baseline.dict"
        if not dict_path.is_file():
            continue
        entries = _iter_mdf_blocks(dict_path)
        pending = sum(1 for e in entries if _GLOSS_PENDING_MARKER in e["rem"])
        total = len(entries)
        summary[lang] = {
            "total": total,
            "curated": total - pending,
            "gloss_pending": pending,
        }
    return summary


def export_gloss_pending(
    lang: str,
    top_n: int = 200,
    baselines_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Extract top-frequency ``gloss-pending`` stems for ``lang`` with example sentences."""
    root = _baselines_root(baselines_dir)
    dict_path = root / lang / "baseline.dict"
    if not dict_path.is_file():
        raise FileNotFoundError(f"Baseline dictionary not found: {dict_path}")
    rows: List[Dict[str, Any]] = []
    for entry in _iter_mdf_blocks(dict_path):
        if _GLOSS_PENDING_MARKER not in entry["rem"]:
            continue
        m = _FREQ_RE.search(entry["rem"])
        freq = int(m.group(1)) if m else 0
        rows.append(
            {
                "lx": entry["lx"],
                "ps": entry["ps"],
                "freq": freq,
                "xv": entry["xv"],
            }
        )
    rows.sort(key=lambda r: (-r["freq"], r["lx"]))
    return rows[: max(0, int(top_n))]
