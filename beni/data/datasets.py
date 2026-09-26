# Dataset loader for Sebeni Alignment

import csv
import copy
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from beni.core.srl.config import DataConfig
from beni.core import Sentence
from beni.utils import config as cfg

try:
    from datasets import load_dataset, Dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    from langdetect import detect as _detect_lang
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

_TEXT_ALIASES = ("text", "sentence", "content", "raw_text", "utt", "question", "doc")
_LANG_ALIASES = ("lang", "language", "lang_id", "lang_code", "lang_tag", "locale")
_FILE_EXTS = {"txt", "text", "json", "jsonl", "ndjson", "csv", "tsv"}


def build_dabax_reference(text: str, lang: str) -> dict:
    """Build ideal completion JSON from the active DabaX checkpoint."""
    from beni.core import sentence_to_completion_json
    from beni.core.language import Language
    from beni.core.morphotactic.dabax import get_dabax

    group = Language.from_code(lang or "bam").group_code
    try:
        sentences = get_dabax(group, process=True).loader(text) or []
    except Exception:
        return {}
    if not sentences:
        return {}
    return sentence_to_completion_json(sentences[0])


EXPERIMENT_JSONL_NAME = "dataset_300_samples.jsonl"
EVAL_SPLIT_NAME = "test.json"


def assert_train_source(path: Union[str, Path]) -> Path:
    """Refuse the held-out eval file as a distillation or training source."""
    source = Path(path)
    if source.name == EVAL_SPLIT_NAME:
        raise ValueError("test.json is evaluation-only and cannot be used for distillation or training")
    return source


def load_experiment_records(
    languages: Optional[Sequence[str]] = None,
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load the experiment train jsonl with the file-only language remap.

    ``mlq`` → ``kao``, ``hsy`` → ``mey``, ``seq`` → ``spp`` happen here, before
    group-code resolution. ``bbo`` rows are kept only when ``bbo`` is an
    allowed language. This function never reads ``test.json``.
    """
    from beni.core.language import OUTLIER_CODE, experiment_group_code

    source = assert_train_source(path or (cfg.DATA_DIR / "raw" / EXPERIMENT_JSONL_NAME))
    allowed = None
    if languages:
        from beni.core.language import resolve_scope

        allowed = set(resolve_scope(languages).languages)
    records: List[Dict[str, str]] = []
    counts: Dict[str, int] = {}
    dropped_bbo = 0
    raw_counts: Dict[str, int] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = str(row.get("lang") or row.get("language") or "").strip().lower()
            raw_counts[raw] = raw_counts.get(raw, 0) + 1
            group = experiment_group_code(raw)
            if group == OUTLIER_CODE and (allowed is None or OUTLIER_CODE not in allowed):
                dropped_bbo += 1
                continue
            if allowed is not None and group not in allowed:
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            records.append({"text": text, "lang": group})
            counts[group] = counts.get(group, 0) + 1
    return {
        "records": records,
        "counts": counts,
        "raw_counts": raw_counts,
        "dropped_bbo": dropped_bbo,
        "source": str(source),
    }


def corrupt_completion(reference: dict) -> dict:
    """Create a legal but morphologically wrong negative completion."""
    rejected = copy.deepcopy(reference)
    tokens = rejected.get("tokens") or []
    for token in tokens:
        try:
            if int(token.get("stage", -1)) == -1:
                token["stage"] = 1
                return rejected
        except (TypeError, ValueError):
            continue
    if tokens:
        tokens[0]["stage"] = -1 if str(tokens[0].get("stage")) != "-1" else 1
        tokens[0]["analyses"] = []
    else:
        rejected["lang"] = "und"
    return rejected


def _split_paragraphs(blob: Any) -> List[str]:
    """Split a packaged test blob into non-empty paragraphs (blank lines, else lines)."""
    text = str(blob or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n")] if "\n\n" in text else [p.strip() for p in text.split("\n")]
    return [p for p in parts if p]

def _get_path(mapping: Dict[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current

def _pick(mapping: Dict[str, Any], key: Optional[str], aliases: Sequence[str]) -> Any:
    for candidate in filter(None, ([key] if key else []) + list(aliases)):
        if (value := _get_path(mapping, candidate)) not in (None, ""):
            return value
    return None

class SebeniDataLoader:
    """Data loader for Sebeni Alignment"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.sentences: List[Sentence] = []
        self.skipped: int = 0
        self._records_cache: Optional[List[Dict[str, str]]] = None

    def __len__(self) -> int:
        return len(self.sentences)

    def __iter__(self) -> Iterator[Dict[str, str]]:
        return iter(self.records)

    @property
    def records(self) -> List[Dict[str, str]]:
        if self._records_cache is None:
            self._records_cache = [{"text": s.text, "lang": s.lang} for s in self.sentences]
        return self._records_cache

    def reset(self) -> "SebeniDataLoader":
        self.sentences.clear()
        self._records_cache = None
        self.skipped = 0
        return self

    def load(
        self, 
        source: Union[str, Path, Sequence[Union[str, Path]]], 
        *, 
        source_type: str = "auto", 
        lang: Optional[str] = None, 
        text_key: Optional[str] = None, 
        lang_key: Optional[str] = None, 
        split: Optional[str] = None, 
        config_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        if self.config.clear_on_load:
            self.reset()
            
        sources = [source] if isinstance(source, (str, Path)) else list(source)
        for src in sources:
            if self._resolve_source_type(src, source_type) == "file":
                for path in self._expand_path(src):
                    self._load_file(Path(path), lang, text_key, lang_key)
            else:
                self._load_hf(str(src), lang, text_key, lang_key, split, config_name)
        return self.records

    @staticmethod
    def _resolve_source_type(src: Union[str, Path], source_type: str) -> str:
        if source_type in ("file", "hf"): 
            return source_type
        if source_type != "auto": 
            raise ValueError(f"Invalid source_type: {source_type}")
        if Path(src).exists() or any(ch in str(src) for ch in "*?["): 
            return "file"
        return "hf"

    @staticmethod
    def _expand_path(src: Union[str, Path]) -> List[Path]:
        p = Path(src)
        if p.is_dir():
            if not (found := sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower().lstrip(".") in _FILE_EXTS)):
                raise ValueError("No supported files found.")
            return found
        if p.exists(): 
            return [p]
        if matches := [m for m in sorted(Path(x) for x in glob.glob(str(src))) if m.is_file()]: 
            return matches
        raise FileNotFoundError(f"No match for '{src}'")

    def _load_file(self, path: Path, lang: Optional[str], text_key: Optional[str], lang_key: Optional[str]) -> None:
        ext = path.suffix.lower()
        if ext in (".txt", ".text"):
            self._load_plain_text(path, lang)
        elif ext in (".jsonl", ".ndjson"):
            self._load_json_file(path, lang, text_key, lang_key, jsonl=True)
        elif ext == ".json":
            self._load_json_file(path, lang, text_key, lang_key, jsonl=False)
        elif ext in (".csv", ".tsv"):
            self._load_csv(path, lang, text_key, lang_key)
        else:
            raise ValueError(f"Unsupported file type '{ext}' for {path}")

    def _load_plain_text(self, path: Path, lang: Optional[str]) -> None:
        file_lang = self._resolve_lang(lang, path=path)
        detect = (file_lang is None and self.config.auto_detect_lang and _LANGDETECT_AVAILABLE)
        
        with open(path, "r", encoding=self.config.encoding, errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                text = line.strip() if self.config.strip_text else line.rstrip("\r\n")
                if self.config.skip_empty and not text:
                    continue
                    
                line_lang = file_lang
                if detect and text:
                    try:
                        line_lang = _detect_lang(text).lower()
                    except Exception:
                        pass
                
                self._append(text, line_lang, path, lineno)

    def _load_json_file(self, path: Path, lang: Optional[str], text_key: Optional[str], lang_key: Optional[str], jsonl: bool) -> None:
        text_key = text_key or self.config.file_text_key
        lang_key = lang_key or self.config.file_lang_key
        with open(path, "r", encoding=self.config.encoding, errors="replace") as fh:
            if jsonl:
                for lineno, line in enumerate(fh, 1):
                    if line := line.strip():
                        self._ingest_json(json.loads(line), path, lang, text_key, lang_key, lineno=lineno)
            else:
                self._ingest_json(json.load(fh), path, lang, text_key, lang_key)

    def _ingest_json(self, value: Any, path: Path, lang: Optional[str], text_key: str, lang_key: str, found_lang: Optional[str]=None, lineno: Optional[int]=None) -> None:
        if isinstance(value, str):
            self._append(value, self._resolve_lang(lang, found_lang, path=path), path, lineno)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._ingest_json(item, path, lang, text_key, lang_key, found_lang, lineno)
        elif isinstance(value, dict):
            if (text := _pick(value, text_key, _TEXT_ALIASES)) is not None:
                resolved = self._resolve_lang(lang, _pick(value, lang_key, _LANG_ALIASES) or found_lang, path=path)
                self._append(text, resolved, path, lineno)
            else:
                for item in value.values():
                    self._ingest_json(item, path, lang, text_key, lang_key, found_lang, lineno)
        elif value is not None:
            self.skipped += 1

    def _load_csv(self, path: Path, lang: Optional[str], text_key: Optional[str], lang_key: Optional[str]) -> None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, "r", newline="", encoding=self.config.encoding, errors="replace") as fh:
            for row in csv.DictReader(fh, delimiter=delimiter):
                if text := _pick(row, text_key, _TEXT_ALIASES):
                    self._append(text, self._resolve_lang(lang, _pick(row, lang_key, _LANG_ALIASES), path=path), path)

    def _load_hf(self, name: str, lang: Optional[str], text_key: Optional[str], lang_key: Optional[str], split: Optional[str], config_name: Optional[str]) -> None:
        kwargs = {"split": split or self.config.hf_split}
        if config_name:
            kwargs["name"] = config_name
        if self.config.hf_streaming:
            kwargs["streaming"] = True
        if self.config.hf_kwargs:
            kwargs.update(self.config.hf_kwargs)
        
        ds = load_dataset(name, **kwargs)
        if hasattr(ds, "column_names"):
            self._ingest_hf_part(ds, name, lang, text_key, lang_key)
        else:
            for part in ds.values():
                self._ingest_hf_part(part, name, lang, text_key, lang_key)

    def _ingest_hf_part(
        self,
        ds: Any,
        name: str,
        lang: Optional[str],
        text_key: Optional[str],
        lang_key: Optional[str] = None,
    ) -> None:
        text_key = text_key or self.config.hf_text_key or "text"
        lang_key = lang_key or self.config.hf_lang_key
        for row in ds:
            found = _pick(row, lang_key, _LANG_ALIASES) if isinstance(row, dict) else None
            text = _get_path(row, text_key) if isinstance(row, dict) else None
            self._append(text, self._resolve_lang(lang, found), name)

    def _resolve_lang(self, explicit: Optional[str], found: Optional[str] = None, path: Optional[Path] = None) -> Optional[str]:
        for candidate in (explicit, found):
            if candidate and str(candidate).strip():
                return str(candidate).strip().lower()
                
        if path and self.config.infer_lang_from_filename:
            tokens = [t for t in re.split(r"[^A-Za-z0-9]+", path.name) if t]
            if tokens and tokens[-1].lower() in _FILE_EXTS: 
                tokens = tokens[:-1]
            if tokens and re.fullmatch(r"[a-z]{2,3}", tokens[-1].lower()):
                return tokens[-1].lower()
                
        return self.config.default_lang.lower() if self.config.default_lang else None

    def _clean_text(self, text: Any) -> Optional[str]:
        if text is None: 
            return None
        text = str(text).strip() if self.config.strip_text else str(text)
        if self.config.skip_empty and not text: 
            return None
        return text[: self.config.max_text_len] if self.config.max_text_len else text

    def _append(self, text: Any, lang: Optional[str], source: Optional[Any] = None, lineno: Optional[int] = None) -> bool:
        if (text := self._clean_text(text)) is None:
            self.skipped += 1
            return False
        from beni.core.language import Language

        resolved = Language.from_code(lang or self.config.default_lang or "bam").group_code
        if self.config.languages:
            allowed = set(Language.group_codes(self.config.languages))
            if resolved not in allowed:
                self.skipped += 1
                return False
        self.sentences.append(Sentence(text=text, lang=resolved))
        self._records_cache = None
        return True

    def load_sebeni(self, data_dir: Union[str, Path] = cfg.DATA_DIR, split: str = "train"):
        """Load packaged Sebeni splits.

        ``split="train"`` reads ``data/raw/*.txt`` (language = filename stem via
        ``Language.from_code``; ``mlq`` → ``mku``). ``split="test"`` reads
        ``data/test.json`` (``{lang: text}``) and splits paragraphs.
        """
        from beni.core.language import Language

        data_dir = Path(data_dir)
        if split == "train":
            source = data_dir / "raw" if (data_dir / "raw").is_dir() else data_dir
            if source == cfg.DATA_DIR:
                source = cfg.DATA_DIR / "raw"
            if not source.is_dir():
                raise FileNotFoundError(f"Packaged train split not found: {source}")
            for path in sorted(p for p in source.iterdir() if p.is_file()):
                if path.suffix.lower().lstrip(".") not in {"txt", "text"}:
                    continue
                lang = Language.from_code(path.stem).group_code
                self._load_plain_text(path, lang)
        else:
            source = data_dir / "test.json"
            if not source.is_file():
                source = cfg.DATA_DIR / "test.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            for lang_key, blob in payload.items():
                lang = Language.from_code(str(lang_key)).group_code
                for para in _split_paragraphs(blob):
                    self._append(para, lang)
        return self.records

    def ds(self, ds: Any = None, scheme: Optional[str] = None) -> Dataset:
        scheme = scheme or self.config.scheme or "completion"
        if scheme == "preference":
            return self._preference_ds(ds)
        return self._completion_ds(ds)

    def _completion_ds(self, ds: Any = None) -> Dataset:
        from beni.core.srl.config import SRLGrpoPrompt
        dataset = {"prompt": [], "language": [], "reference": []}

        langs = list(self.config.languages or [])
        if not langs and self.config.default_lang:
            langs = [self.config.default_lang]
        prompt = SRLGrpoPrompt(languages=langs).prompt()
        items = ds if ds is not None else self.records

        for item in items:
            if isinstance(item, dict):
                text = item.get("text", "")
                lang = item.get("lang", item.get("language", self.config.default_lang))
                reference = item.get("reference", {})
            else:
                text = getattr(item, "text", "")
                lang = getattr(item, "lang", self.config.default_lang)
                reference = {}

            if not reference and text:
                reference = build_dabax_reference(text, lang or self.config.default_lang or "bam")

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]

            dataset["prompt"].append(messages)
            dataset["language"].append(lang)
            dataset["reference"].append(reference)

        return Dataset.from_dict(dataset)

    def _preference_ds(self, ds: Any = None) -> Dataset:
        items = ds if ds is not None else self.records
        langs = list(self.config.languages or [])
        if not langs and self.config.default_lang:
            langs = [self.config.default_lang]
        return rank_group_to_preference(items, languages=langs)

    def format_to_dataset(self, sentences: Any = None) -> Dataset:
        return self.ds(sentences)


def rank_group_to_preference(
    records: Sequence[Any],
    scheme_prompt: bool = True,
    languages: Optional[Sequence[str]] = None,
) -> "Dataset":
    """Rank SAMPG group completions (DabaX / reward scores) into chosen/rejected.

    Each record may supply ``chosen``/``rejected`` directly, or ``completions`` plus
    ``scores`` (higher is better). Offline DPO/APO still come from ranking those groups.
    """
    from beni.core.srl.config import SRLGrpoPrompt

    system = SRLGrpoPrompt(languages=list(languages) if languages else None).prompt() if scheme_prompt else ""
    rows = {"prompt": [], "chosen": [], "rejected": [], "language": []}
    for item in records or []:
        if not isinstance(item, dict):
            continue
        lang = item.get("lang") or item.get("language") or ""
        user = item.get("text") or item.get("prompt") or ""
        chosen = item.get("chosen")
        rejected = item.get("rejected")
        completions = item.get("completions") or []
        scores = item.get("scores") or []
        if chosen is None and completions and scores and len(completions) == len(scores):
            best_i = max(range(len(scores)), key=lambda i: scores[i])
            worst_i = min(range(len(scores)), key=lambda i: scores[i])
            chosen, rejected = completions[best_i], completions[worst_i]
        if chosen is None:
            ref = item.get("reference")
            if not ref:
                ref = build_dabax_reference(user, lang)
            chosen = ref
        if rejected is None:
            if isinstance(chosen, str):
                try:
                    chosen_obj = json.loads(chosen)
                except (TypeError, json.JSONDecodeError):
                    chosen_obj = {}
            else:
                chosen_obj = chosen or {}
            rejected = corrupt_completion(chosen_obj)
        if isinstance(chosen, dict):
            chosen = json.dumps(chosen, ensure_ascii=False)
        if isinstance(rejected, dict):
            rejected = json.dumps(rejected, ensure_ascii=False)
        rows["prompt"].append(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            if scheme_prompt
            else user
        )
        rows["chosen"].append(str(chosen))
        rows["rejected"].append(str(rejected))
        rows["language"].append(lang)
    return Dataset.from_dict(rows)


"""
[('Bambara', 'bam'),
 ('Bomu', 'bmq'),
 ('Bozo', 'boz'),
 ('Dogon', 'dtm'),
 ('Fula', 'ful'),
 ('Hassaniyya', 'mey'),
 ('Maninka', 'mku'),
 ('Khassonke (Xaasongaxango)', 'kao'),
 ('Mamara', 'myk'),
 ('Senoufo', 'spp'),
 ('Songhay', 'ses'),
 ('Soninke', 'snk'),
 ('Tamasheq', 'taq')]
"""
