"""
Configuration management for Sɛbɛni - SLM Alignment Framework
"""

import os
import json
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union

PACKAGE_NAME = "Sɛbɛni v0.1 - SLM Alignment Framework"
logger = logging.getLogger(__name__)


def load_dotenv(paths=None) -> list:
    """Load simple KEY=VALUE files without overriding the process environment."""
    candidates = list(paths or [])
    explicit = os.getenv("SEBENI_ENV")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    loaded = []
    seen = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ.setdefault(key, value)
        loaded.append(path)
    return loaded


# Import-time convenience for library users; YAML loading also checks its own directory/workdir.
load_dotenv([Path.cwd() / ".env"])

# API Keys
GOOGLE_API = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID") or os.getenv("GOOGLE_PROJECT")
GOOGLE_LOCATION = os.getenv("GOOGLE_LOCATION") or os.getenv("GOOGLE_REGION") or "us-central1"
OPENAI_API = os.getenv("OPENAI_API_KEY")
GROQ_API = os.getenv("GROQ_API_KEY")
TOGETHER_API = os.getenv("TOGETHER_API_KEY")

PROVIDER_API_KEYS = {
    "google": GOOGLE_API,
    "gemini": GOOGLE_API,
    "openai": OPENAI_API,
    "groq": GROQ_API,
    "together": TOGETHER_API,
}

# Paths
HOME_DIR = Path.home()
BASE_DIR = Path(__file__).parent.parent
DEFAULT_WORKING_DIR = Path.home() / ".sebeni"
DATA_DIR = BASE_DIR / "data"

# Packaged Maninka resources live under baselines/mlq/; metadata group_code is mku.
PACKAGED_BASELINE_ALIASES = {"mku": "mlq"}

WORKING_DIR = DEFAULT_WORKING_DIR
CHECKPOINTS_DIR = WORKING_DIR / "runs"
MODEL_DIR = WORKING_DIR / "models"
WD_DATA = WORKING_DIR / "data"
EXP_DIR = WORKING_DIR / "exp"
RUNTIME_DIR = WORKING_DIR / "runtime"

# Default model configurations
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 10
TEMPERATURE = 0.7

# Token limits & Cache Config
MAX_TOKENS_PER_REQUEST = 1000000
CACHE_TTL = "3600s"
CACHE_NAME = "sebeni_cache"

SCRATCH_GRAM = """# sebeni-scratch-bootstrap
# Empty Daba-compatible grammar. Distiller bootstrap fills this in.
plan
for token:
stage 0 apply lookup
return if parsed
"""

SCRATCH_DICT = """\\_sh v3.0  400  MDF 4.0

\\lang {lang}
\\name sebeni-scratch
\\ver 0.0.0

\\lx _
\\ps n
\\ge scratch
"""

# Load resources
MDF_FIELDS = json.load(open(BASE_DIR / "utils/res/mdf_fields.json"))
LANGUAGES_METADATA = json.load(open(BASE_DIR / "utils/res/language_metadata.json"))['languages']
ISO_CODES = json.load(open(BASE_DIR / "utils/res/iso_code_639_3.json"))


class Workdir:
    """Relocatable working directory. Layout: data/, models/, runs/, exp/, runtime/."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser().resolve()

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def exp(self) -> Path:
        return self.root / "exp"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def baselines(self) -> Path:
        return self.data / "baselines"

    def ensure(self) -> "Workdir":
        for path in (self.root, self.runs, self.models, self.data, self.exp, self.runtime, self.baselines):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def __fspath__(self) -> str:
        return str(self.root)

    def __str__(self) -> str:
        return str(self.root)


_active_workdir = Workdir(DEFAULT_WORKING_DIR)


def get_workdir() -> Workdir:
    """Return the active working directory (does not create it)."""
    return _active_workdir


def set_working_dir(path: Union[str, Path], ensure: bool = True) -> Workdir:
    """Set the active working directory and rebind module path constants.

    Call once at CLI/config load, before Distiller or trainers construct paths.

    Parameters
    ----------
    path : str or Path
        Absolute or relative directory. Relative paths are resolved against cwd.
    ensure : bool
        Create the standard layout if missing.
    """
    global WORKING_DIR, CHECKPOINTS_DIR, MODEL_DIR, WD_DATA, EXP_DIR, RUNTIME_DIR, _active_workdir
    _active_workdir = Workdir(path)
    if ensure:
        _active_workdir.ensure()
    WORKING_DIR = _active_workdir.root
    CHECKPOINTS_DIR = _active_workdir.runs
    MODEL_DIR = _active_workdir.models
    WD_DATA = _active_workdir.data
    EXP_DIR = _active_workdir.exp
    RUNTIME_DIR = _active_workdir.runtime
    return _active_workdir


def resolve_working_dir(
    cli_path: Optional[Union[str, Path]] = None,
    yaml_path: Optional[Union[str, Path]] = None,
    config_file_dir: Optional[Union[str, Path]] = None,
    ensure: bool = True,
) -> Workdir:
    """Resolve workdir. Later wins only if set: default, env, YAML, CLI.

    Relative YAML paths resolve against the config file's directory.
    Relative CLI ``-w`` resolves against cwd.
    """
    root = DEFAULT_WORKING_DIR

    env = os.getenv("SEBENI_WORKING_DIR") or os.getenv("SEBENI_HOME")
    if env:
        root = Path(env).expanduser()

    if yaml_path:
        yp = Path(yaml_path).expanduser()
        if not yp.is_absolute() and config_file_dir is not None:
            yp = Path(config_file_dir) / yp
        root = yp

    if cli_path:
        root = Path(cli_path).expanduser()

    if not Path(root).is_absolute():
        root = Path.cwd() / root

    return set_working_dir(root, ensure=ensure)


def resolve_packaged_baseline_dir(code: str) -> Optional[Path]:
    """Return the packaged baseline directory for a language/group code.

    MKU (group) aliases to on-disk ``baselines/mlq/``.
    """
    needle = str(code or "").strip().lower()
    group = get_group_code(needle) or needle
    base = DATA_DIR / "baselines"
    candidates = [
        group,
        needle,
        PACKAGED_BASELINE_ALIASES.get(group, ""),
        PACKAGED_BASELINE_ALIASES.get(needle, ""),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = base / candidate
        if path.exists():
            return path
    return None


def clean_folder_tree(path):
    if path.exists():
        shutil.rmtree(path)


def get_language_name(code: str) -> str:
    lang = get_language_metadata(code)
    return lang.get('language', code)


def get_language_metadata(code: str, default: Dict = {}) -> Dict:
    needle = str(code or "").strip().lower()
    for lang in LANGUAGES_METADATA:
        if str(lang.get('group_code', '')).lower() == needle:
            return lang
        for variant in lang.get('variants', []):
            if str(variant.get('code', '')).strip().lower() == needle:
                return lang
    return default


def get_group_code(code: str) -> str:
    lang = get_language_metadata(code)
    if lang:
        return lang.get('group_code', code)
    return code


def get_language_iso(code: str = 'xyz') -> Dict:
    lang = [i for i in ISO_CODES if i['code'] == code]
    return lang[0] if lang else {}


def provider_api_key(provider: str) -> Optional[str]:
    """Environment API key for a distillation provider name."""
    env_names = {
        "google": ("GOOGLE_API_KEY", "GOOGLE_API"),
        "gemini": ("GOOGLE_API_KEY", "GOOGLE_API"),
        "openai": ("OPENAI_API_KEY",),
        "groq": ("GROQ_API_KEY",),
        "together": ("TOGETHER_API_KEY",),
    }
    for name in env_names.get(str(provider or "").lower(), ()):
        value = os.getenv(name)
        if value:
            return value
    return PROVIDER_API_KEYS.get(str(provider or "").lower())


_ADC_WELL_KNOWN = (
    Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
)


def google_adc_path() -> Optional[Path]:
    """ADC JSON from ``GOOGLE_APPLICATION_CREDENTIALS`` or the gcloud well-known file."""
    explicit = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    if _ADC_WELL_KNOWN.is_file():
        return _ADC_WELL_KNOWN
    return None


def google_project_id() -> Optional[str]:
    """GCP project for Vertex: env first, then ADC ``quota_project_id`` / ``project_id``."""
    for name in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT_ID", "GOOGLE_PROJECT"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    path = google_adc_path()
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("quota_project_id", "project_id"):
        value = data.get(key)
        if value:
            return str(value)
    return None


def google_location() -> str:
    for name in ("GOOGLE_CLOUD_LOCATION", "GOOGLE_LOCATION", "GOOGLE_REGION"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return GOOGLE_LOCATION or "us-central1"


def use_google_vertex(vertex: Optional[bool], api_key: Optional[str] = None) -> bool:
    """Select Vertex/ADC vs Gemini Developer API (AI Studio).

    ``True`` always Vertex. ``False`` always the Studio API-key client.
    ``None`` auto-selects Vertex when ADC project or credentials exist, even if
    ``GOOGLE_API_KEY`` is also set. A leftover Studio key must not silently
    bill AI Studio prepaid credits.
    """
    del api_key
    if vertex is True:
        return True
    if vertex is False:
        return False
    return bool(google_project_id() or google_adc_path())


def scratch_gram() -> str:
    """Daba-compatible empty grammar stub for languages with no packaged baseline."""
    return SCRATCH_GRAM


def scratch_dict(lang: str) -> str:
    """Daba-compatible empty dictionary stub."""
    return SCRATCH_DICT.format(lang=lang or "und")


@dataclass
class Config:
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    working_dir: Path = field(default_factory=lambda: get_workdir().root)
    prompts_dir: Optional[Path] = None
    
    default_provider: str = "google"
    default_model: str = DEFAULT_MODEL
    temperature: float = TEMPERATURE
    max_retries: int = 3
    retry_wait: int = 3
    
    max_chars_per_file: int = 4000
    max_total_chars: int = 12000
    cache_ttl: str = CACHE_TTL
    
    language_code: str = "bam"
    baseline_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    sample_files: List[Path] = field(init=False, default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir)
        
        if self.prompts_dir is None:
            self.prompts_dir = self.base_dir / "utils" / "prompts"
        
        self.baseline_dir = self.data_dir / "baselines"
        self.checkpoint_dir = self.working_dir / "runs"
        self.sample_files = list(self.data_dir.glob("samples/*"))
        
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def format_mdf_fields(self) -> str:
        return '\n'.join([
            f"- \\{code} : {info.get('name', code)} - {info.get('description', '')}"
            for code, info in MDF_FIELDS.items()])


def copy_file_if_exists(source_dir: str, dest_dir: str, filename: str) -> None:
    """
    Copy a file from source_dir to dest_dir.
    Assumes (and checks) that the file does exist in source_dir —
    raises FileNotFoundError in that case.
    """
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)

    if os.path.isfile(dest_path):
        print(f"File '{filename}' already exists in dest_dir: {dest_dir}")
        return

    if not os.path.isfile(source_path):
        raise FileNotFoundError(
            f"File '{filename}' does not exist in source_dir: {source_dir}"
        )

    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    shutil.copy2(source_path, dest_path)  # copy2 preserves metadata
    print(f"Copied: {source_path} => {dest_path}")
