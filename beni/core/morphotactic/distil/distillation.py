import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
import logging
import tempfile
from typing import Callable, Iterable, List, Optional, Union, Dict, Any
from beni.utils import config as cfg
from beni.utils.hf_lang import get_language_metadata as lg
from beni.core.language import Language
from beni.core.morphotactic.distil.providers import create_provider
from beni.core.compute.metrics import MorphologyScorer

REF_SAMPLES_DIR = cfg.DATA_DIR / "samples"
LANG_BASELINE = cfg.DATA_DIR / "baselines"
SCRATCH_MARKER = "sebeni-scratch-bootstrap"

logger = logging.getLogger(__name__)


@dataclass
class DistillProposal:
    """Candidate {G, D} from Distiller, scored but not yet written."""

    gram_text: str
    dict_text: str
    phi: float
    phi_prime: float
    parseable: bool
    first_create: bool
    language_ok: bool = True
    mer: Optional[float] = None


class Distiller(object):
    """SAMPG Distiller: checkpoints, scratch bootstrap, promote."""

    def __init__(
        self,
        lang_code: str,
        provider: Optional[str] = None,
        backend: Optional[str] = None,
        model: str = "gemma-4-26b-a4b-it",
        api_key: Optional[str] = None,
        working_dir: Union[None, str] = None,
        vertex: Optional[bool] = None,
        location: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: Optional[str] = None,
        gguf_path: Optional[str] = None,
        n_ctx: int = 4096,
        max_input_chars: int = 8000,
    ):
        """Initialize Distiller for a language/group code.

        Parameters
        ----------
        lang_code : str
            ISO or Sebeni group code. Maninka ``mku`` aliases to packaged ``mlq/``.
        working_dir : str, optional
            Root workdir. Baselines live under ``{root}/data/baselines/{group}/``.
        """
        identity = Language.from_code(lang_code)
        self.lang = identity.group_code
        self.language = identity
        self.language_meta = self._valid_language(lang_code)
        self.provider_name = str(backend or provider or "algorithmic").lower()
        self.max_input_chars = int(max_input_chars)
        self.location = location
        self.project_id = project_id
        self.provider = None
        if self.provider_name != "algorithmic":
            key = api_key or cfg.provider_api_key(self.provider_name)
            use_vertex = False
            extra = {
                "language": self.lang,
                "base_url": base_url,
                "gguf_path": gguf_path,
                "n_ctx": n_ctx,
            }
            if self.provider_name in {"google", "gemini"}:
                use_vertex = cfg.use_google_vertex(vertex, key)
                extra["vertex"] = use_vertex
                if use_vertex:
                    extra["project_id"] = project_id or cfg.google_project_id()
                    extra["region"] = location or cfg.google_location()
                    if key:
                        logger.warning(
                            "GOOGLE_API_KEY is set but Vertex/ADC is selected; "
                            "the Gemini Developer API key will not be used. "
                            "Set distillation.vertex: false to use AI Studio."
                        )
            self.provider = create_provider(
                self.provider_name, api_key=key, model=model, **extra
            )

        packaged = cfg.resolve_packaged_baseline_dir(self.lang)
        self.lang_baseline = Path(packaged) if packaged else None
        self.sample_dir = REF_SAMPLES_DIR
        root = Path(working_dir) if working_dir else cfg.get_workdir().root
        self.working_root = Path(root)
        self.baselines_dir = self.working_root / "data" / "baselines" / self.lang
        latest = self.__get_latest_ckpt()
        self.gram_path = latest["gram"] or (self.baselines_dir / "baseline.gram")
        self.dict_path = latest["dict"] or (self.baselines_dir / "baseline.dict")
        self.gram_delta = None
        self.dict_delta = None
        self.bootstrapped = False

    def handle_baselines(self):
        """Copy packaged baselines or write scratch stubs under the workdir."""
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        gram_dest = self.baselines_dir / "baseline.gram"
        dict_dest = self.baselines_dir / "baseline.dict"

        if self.lang_baseline and self.lang_baseline.exists():
            for name in ("baseline.gram", "baseline.dict"):
                src = self.lang_baseline / name
                if src.exists() and not (self.baselines_dir / name).exists():
                    cfg.copy_file_if_exists(self.lang_baseline, self.baselines_dir, name)
            self.bootstrapped = False
        elif not gram_dest.exists() or not dict_dest.exists():
            gram_dest.write_text(cfg.scratch_gram(), encoding="utf-8")
            dict_dest.write_text(cfg.scratch_dict(self.lang), encoding="utf-8")
            self.bootstrapped = True
            logger.info("Wrote scratch baselines for '%s' under %s", self.lang, self.baselines_dir)

        latest = self.__get_latest_ckpt()
        self.gram_path = latest["gram"] or gram_dest
        self.dict_path = latest["dict"] or dict_dest
        return self.baselines_dir
        
    def _valid_language(self, code: str) -> Union[bool, Dict]:
        """ Validate language code """
        lang = cfg.get_language_metadata(code)

        if(not lang):
            lang = cfg.get_language_iso(code)
            if( not lang):
                # TODO: Change to normal Logging
                raise Exception(
                    f"Language code '{code}' is not a iso_639_3 standard code. "
                    "Manually edit language metadata to include your language with self define 3-char code."
                )
            msg = f"Language '{code}' is not Standard Sebeni 'code', it will be treated as '{lang['name']}'"
            # warnings.warn(msg, UserWarning, stacklevel=1, source="Distiller._valid_language")
            logger.warning(msg)
            lang = lg(code)
        return lang

    def distill_state(self, baseline_dir: Union[str, Path]=None):
        baseline_dir = self.handle_baselines() if not baseline_dir else baseline_dir
        return self.__get_latest_ckpt()

    def is_scratch(self) -> bool:
        """True when the current grammar is a scratch bootstrap stub."""
        path = Path(self.gram_path) if self.gram_path else self.baselines_dir / "baseline.gram"
        if not path.exists():
            return True
        try:
            return SCRATCH_MARKER in path.read_text(encoding="utf-8")[:400]
        except OSError:
            return True

    def is_first_create(self) -> bool:
        """True until a versioned ``baseline_vN`` checkpoint exists."""
        latest = self.__get_latest_ckpt()
        gram = latest.get("gram")
        if gram is None:
            return True
        return "_v" not in Path(gram).stem

    def checkpoint_id(self) -> str:
        latest = self.__get_latest_ckpt()
        gram = latest.get("gram")
        if gram is None:
            return "none"
        return Path(gram).stem

    def prompt_mode(self) -> str:
        return "bootstrap" if self.is_scratch() else "delta"

    def _load_dabax(self, gram, ldict):
        from beni.core.morphotactic.dabax import get_dabax

        runtime = self.working_root / "runtime"
        return get_dabax(
            self.lang,
            gram=gram,
            ldict=ldict,
            process=True,
            runtime_dir=runtime,
            working_dir=self.working_root,
        )

    def phi_on_texts(self, texts: Iterable[str], gram=None, ldict=None) -> float:
        """Corpus Φ on ``texts`` with the given (or current) gram/dict."""
        gram = gram or self.gram_path
        ldict = ldict or self.dict_path
        texts = [t for t in texts if t]
        if not texts:
            return 0.0
        try:
            dx = self._load_dabax(gram, ldict)
        except Exception as exc:
            logger.warning("DabaX failed to load G,D (%s); Φ=0", exc)
            return 0.0
        sentences = []
        for text in texts:
            try:
                sentences.extend(dx.loader(text) or [])
            except Exception as exc:
                logger.debug("DabaX.loader failed: %s", exc)
        if not sentences:
            return 0.0
        return float(MorphologyScorer().phi_corpus(sentences)["avg"])

    def files_parseable(self, gram, ldict, sample: str = "a") -> bool:
        """First-create / candidate gate: DabaX can load the files."""
        try:
            dx = self._load_dabax(gram, ldict)
            dx.loader(sample)
            return True
        except Exception as exc:
            logger.warning("Candidate gram/dict not parseable: %s", exc)
            return False

    def get_sys_instruction(self, language_meta: Dict, gram_path: Path, dict_path: Path) -> str:
        """Generate system instruction for the distillation process."""
        from beni.core.morphotactic.distil import DistilSysPrompt

        samples = self.default_samples()
        mode = self.prompt_mode()
        # Never send a complete production dictionary to an LLM. The miss report
        # and a bounded format sample are sufficient for optional refinement.
        gram_text = gram_path.read_text(encoding="utf-8") if Path(gram_path).exists() else ""
        dict_text = dict_path.read_text(encoding="utf-8") if Path(dict_path).exists() else ""
        return DistilSysPrompt(
            language=json.dumps(language_meta, indent=2, ensure_ascii=False),
            samples=samples,
            current_gram=gram_text[: self.max_input_chars],
            current_dict=dict_text[: self.max_input_chars],
            mode=mode,
        ).prompt()

    def collect_misses(self, texts: Iterable[str], gram=None, ldict=None) -> List[str]:
        """Return unique DabaX stage -1 surfaces in stable encounter order."""
        try:
            dx = self._load_dabax(gram or self.gram_path, ldict or self.dict_path)
        except Exception as exc:
            logger.warning(
                "DabaX failed to load G, D for miss collection (%s); keeping previous files",
                exc,
            )
            return []
        misses: List[str] = []
        seen = set()
        for text in texts:
            try:
                sentences = dx.loader(text) or []
            except Exception as exc:
                logger.debug("DabaX miss collection failed: %s", exc)
                continue
            for sentence in sentences:
                for token in sentence.tokens:
                    try:
                        unknown = int(token.stage) == -1
                    except (TypeError, ValueError):
                        unknown = False
                    surface = str(token.surface or "").strip()
                    if unknown and surface and surface not in seen:
                        seen.add(surface)
                        misses.append(surface)
        return misses

    @staticmethod
    def _algorithmic_dict(current: Path, misses: Iterable[str]) -> str:
        """Append conservative lookup entries for unseen surfaces."""
        text = current.read_text(encoding="utf-8")
        existing = set(re.findall(r"(?m)^\\lx\s+(.+?)\s*$", text))
        blocks = []
        for surface in misses:
            if surface in existing or "\n" in surface or "\r" in surface:
                continue
            blocks.append(f"\\lx {surface}\n\\ps x\n\\ge auto:{surface}")
        if not blocks:
            return text
        return text.rstrip() + "\n\n" + "\n\n".join(blocks) + "\n"

    def _as_output(self, response: Any):
        from beni.core.morphotactic.distil import DistilOutput

        if isinstance(response, DistilOutput):
            return response
        if isinstance(response, dict):
            try:
                return DistilOutput.model_validate(response)
            except Exception:
                return DistilOutput(
                    sebeni_gram=str(response.get("sebeni_gram", "")),
                    sebeni_dict=str(response.get("sebeni_dict", "")),
                )
        return None

    def _texts_parseable(self, gram_text: str, dict_text: str, sample: str = "a") -> bool:
        """Write candidate texts to temps and run the DabaX parse gate."""
        temp_gram = temp_dict = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gram", mode="w", encoding="utf-8") as gf:
                gf.write(gram_text)
                temp_gram = Path(gf.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dict", mode="w", encoding="utf-8") as df:
                df.write(dict_text)
                temp_dict = Path(df.name)
            return self.files_parseable(temp_gram, temp_dict, sample=sample)
        except Exception as exc:
            logger.warning("Candidate gram/dict not parseable: %s", exc)
            return False
        finally:
            if temp_gram and temp_gram.exists():
                temp_gram.unlink()
            if temp_dict and temp_dict.exists():
                temp_dict.unlink()

    def propose(
        self,
        texts: Union[str, List[str]],
        indicator: bool = False,
        current_phi: Optional[float] = None,
    ) -> Optional[DistillProposal]:
        """Produce candidate G, D and score Φ vs Φ′ without writing."""
        if isinstance(texts, list):
            batch = texts
        else:
            batch = [texts]
        self.handle_baselines()
        try:
            return self._propose_candidate(batch, indicator=indicator, current_phi=current_phi)
        except Exception as exc:
            logger.warning("Distiller propose failed; keeping previous G, D (%s)", exc)
            return None

    def _propose_candidate(
        self,
        batch: List[str],
        indicator: bool = False,
        current_phi: Optional[float] = None,
    ) -> Optional[DistillProposal]:
        from beni.core.morphotactic.distil import DistilUserPrompt

        state = self.__get_latest_ckpt()
        self.gram_path = state["gram"] if state["gram"] else self.gram_path
        self.dict_path = state["dict"] if state["dict"] else self.dict_path
        # Scratch stubs use the parse gate. A copied packaged baseline is not a
        # first create: the first promoted checkpoint still requires Φ′ > Φ.
        first_create = self.is_scratch()
        bootstrap = first_create

        current_gram = Path(self.gram_path)
        current_dict = Path(self.dict_path)
        current_gram_text = current_gram.read_text(encoding="utf-8") if current_gram.exists() else ""
        current_dict_text = current_dict.read_text(encoding="utf-8") if current_dict.exists() else ""
        sample = batch[0] if batch else "a"
        misses = self.collect_misses(batch, current_gram, current_dict)
        if not misses:
            logger.info("Distiller: no DabaX misses to update")
            return None

        if self.provider_name == "algorithmic":
            adjusted_gram = current_gram_text
            adjusted_dict = self._algorithmic_dict(current_dict, misses)
            if not self._texts_parseable(adjusted_gram, adjusted_dict, sample):
                logger.warning(
                    "Algorithmic dictionary candidate not parseable; keeping previous G, D"
                )
                return None
        else:
            miss_report = "\n".join(f"- {surface}" for surface in misses)
            sys_prompt = self.get_sys_instruction(
                language_meta=self.language_meta,
                gram_path=self.gram_path,
                dict_path=self.dict_path,
            )
            cap = getattr(getattr(self.provider, "capability", None), "value", None)
            if not getattr(self.provider, "cache", None) and cap in {"both", "cache"}:
                if hasattr(self.provider, "set_cache_contents"):
                    self.provider.set_cache_contents(
                        self.language_meta, self.gram_path, self.dict_path
                    )
                else:
                    self.provider.create_cache(contents=sys_prompt)
            prompt = DistilUserPrompt(
                f"DabaX miss report for {self.lang}:\n{miss_report}",
                self.gram_delta,
                self.dict_delta,
            ).prompt()
            response = self.provider.generate(
                prompt=prompt, sys_instruct=sys_prompt, indicator=indicator
            )
            output = self._as_output(response)
            if output is None:
                logger.warning("Distiller: empty provider response")
                return None
            gram_delta = output.sebeni_gram or ""
            dict_delta = output.sebeni_dict or ""
            if bootstrap or self.prompt_mode() == "bootstrap":
                adjusted_gram, adjusted_dict = self._bootstrap_parseable_pair(
                    current_gram_text,
                    current_dict_text,
                    gram_delta,
                    dict_delta or self._algorithmic_dict(current_dict, misses),
                    sample,
                )
            else:
                adjusted_gram = current_gram_text
                adjusted_dict = current_dict_text
                if gram_delta.strip():
                    adjusted_gram = self.apply_gram_deltas(
                        current_gram_text,
                        gram_delta,
                        keep_if=lambda candidate: self._texts_parseable(
                            candidate, adjusted_dict, sample
                        ),
                    )
                if dict_delta.strip():
                    adjusted_dict = self.apply_dict_deltas(
                        current_dict_text,
                        dict_delta,
                        keep_if=lambda candidate: self._texts_parseable(
                            adjusted_gram, candidate, sample
                        ),
                    )
                else:
                    algorithmic = self._algorithmic_dict(current_dict, misses)
                    if self._texts_parseable(adjusted_gram, algorithmic, sample):
                        adjusted_dict = algorithmic
            if not self._texts_parseable(adjusted_gram, adjusted_dict, sample):
                logger.warning(
                    "Candidate gram/dict not parseable; reverting to previous G, D"
                )
                return None
            if (
                adjusted_gram == current_gram_text
                and adjusted_dict == current_dict_text
            ):
                logger.info(
                    "Distiller: skipped unparseable [ADD]/[REMOVE] deltas; keeping previous G, D"
                )
                return None

        phi = (
            float(current_phi)
            if current_phi is not None
            else self.phi_on_texts(batch, gram=current_gram, ldict=current_dict)
        )
        parseable = True
        phi_prime = self._phi_on_texts_content(
            batch, adjusted_gram, adjusted_dict, fallback=0.0
        )

        return DistillProposal(
            gram_text=adjusted_gram,
            dict_text=adjusted_dict,
            phi=phi,
            phi_prime=phi_prime,
            parseable=parseable,
            first_create=first_create or bootstrap,
            language_ok=True,
        )

    def _bootstrap_parseable_pair(
        self,
        current_gram: str,
        current_dict: str,
        gram_delta: str,
        dict_delta: str,
        sample: str,
    ) -> tuple:
        """Accept full-file bootstrap dumps only when the pair parses; else keep old."""
        candidate_gram = gram_delta if gram_delta.strip() else current_gram
        candidate_dict = dict_delta if dict_delta.strip() else current_dict
        if self._texts_parseable(candidate_gram, candidate_dict, sample):
            return candidate_gram, candidate_dict
        if gram_delta.strip() and self._texts_parseable(gram_delta, current_dict, sample):
            logger.warning("Bootstrap dictionary not parseable; keeping previous D")
            return gram_delta, current_dict
        if dict_delta.strip() and self._texts_parseable(current_gram, dict_delta, sample):
            logger.warning("Bootstrap grammar not parseable; keeping previous G")
            return current_gram, dict_delta
        logger.warning("Bootstrap gram/dict not parseable; keeping previous G, D")
        return current_gram, current_dict

    def _phi_on_texts_content(
        self,
        batch: List[str],
        gram_text: str,
        dict_text: str,
        fallback: float = 0.0,
    ) -> float:
        temp_gram = temp_dict = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gram", mode="w", encoding="utf-8") as gf:
                gf.write(gram_text)
                temp_gram = Path(gf.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dict", mode="w", encoding="utf-8") as df:
                df.write(dict_text)
                temp_dict = Path(df.name)
            return self.phi_on_texts(batch, gram=temp_gram, ldict=temp_dict)
        except Exception as exc:
            logger.warning("Φ′ scoring failed (%s); treating candidate as no improvement", exc)
            return fallback
        finally:
            if temp_gram and temp_gram.exists():
                temp_gram.unlink()
            if temp_dict and temp_dict.exists():
                temp_dict.unlink()

    def write_checkpoint(self, gram_text: str, dict_text: str):
        """Write ``baseline_vN`` after SafetyGovernor has allowed the promote."""
        return self.update_baselines_with_deltas(gram_text, dict_text)

    def run_distillation(self, text: str, indicator=False):
        """Run Distiller and promote iff SAMPG / SafetyGovernor allow it."""
        from beni.core.safety.governor import SafetyGovernor

        proposal = self.propose(text if isinstance(text, list) else [text], indicator=indicator)
        if proposal is None:
            return None, None
        governor = SafetyGovernor()
        decision = governor.allow_promote(
            proposal.phi,
            proposal.phi_prime,
            parseable=proposal.parseable,
            first_create=proposal.first_create,
            language_ok=proposal.language_ok,
        )
        if not decision.allowed:
            logger.info("Distiller: promote refused (%s)", decision.reason)
            return None, None
        return self.write_checkpoint(proposal.gram_text, proposal.dict_text)

    def run_batch_distillation(self, texts: list, indicator=False):
        """Run distillation on a batch of texts."""
        texts = "\n".join(texts) if isinstance(texts, list) else texts
        result = self.run_distillation(texts, indicator=indicator)
        return result

    def update_baselines_with_deltas(self, gram_delta: str, dict_delta: str):
        """ Update the baselines with the provided grammar and dictionary deltas. """
        g_w = False 
        d_w = False

        new_ckpt_num = self.__next_ckpt()
        new_gram_path = self.baselines_dir / f"baseline_v{new_ckpt_num}.gram"
        new_dict_path = self.baselines_dir / f"baseline_v{new_ckpt_num}.dict"

        new_gram_path.write_text(gram_delta, encoding="utf-8")
        if(new_gram_path.exists()):
            g_w = True
            print(f"Updated grammar baseline to {new_gram_path}")
        
        new_dict_path.write_text(dict_delta, encoding="utf-8")
        if(new_dict_path.exists()):
            d_w = True
            print(f"Updated dictionary baseline to {new_dict_path}")

        if(g_w and d_w):
            logger.info(f"Updated grammar and dictionary baselines to {new_gram_path} and {new_dict_path}")
            self.gram_path = new_gram_path
            self.dict_path = new_dict_path

            return new_gram_path, new_dict_path

        return None, None

    def __get_latest_ckpt(self) -> Union[str, Path]:
        """ Get the latest checkpoint file """
        state = {'gram': None, 'dict': None}

        highest = 0

        suffixes = ['.gram', '.dict']
        baseline = "baseline"

        if(self.baselines_dir.exists()):
            for f in self.baselines_dir.iterdir():
                if(f.suffix in suffixes):
                    ckpt_ext = f.suffix.strip('.')
                    parts = str(f).split("_v")
                    if(len(parts) >  1):
                        try:
                            num = int(parts[1].replace(f.suffix, ""))
                            if(num >= highest):
                                highest = num
                                state[ckpt_ext] = f
                                continue
                        except ValueError:
                            continue
                    else:
                        if(f.stem == baseline and not highest):
                            state[ckpt_ext] = f
        else:    
            print("Blank Baselines: LLM only Parsing")
        
        return state

    def __next_ckpt(self):
        """ Get the version count of next ckpt  """
        latest = self.__get_latest_ckpt()
        gram = latest['gram']
        ldict = latest['dict']

        if(not self.baselines_dir.exists()):
            return 1

        if(not gram or not ldict):
            return 1

        gparts = gram.stem.split("_v")
        dparts = ldict.stem.split("_v")

        if(len(gparts) > 1 and len(dparts) > 1):
            try:
                gnum = int(gparts[1])
                dnum = int(dparts[1])

                if(gnum != dnum):
                    # TODO: Logging module
                    print(f"Warning: Checkpoint version mismatch: {gram} vs {ldict}")

                return max(gnum, dnum) + 1
            except ValueError:
                return 1
        else:
            return 1

    def pp(self):
        print(self.__next_ckpt())


    def default_samples(self):
        """ Load samples and reference files to guide LLM response """
        samples = {}

        if(self.sample_dir.exists()):
            for f in os.listdir(self.sample_dir):
                if(f.endswith('.gram') or f.endswith('.dict') or f.endswith('.guide')):
                    samples[f] = open(self.sample_dir/f, "r").read()

        return samples

    def apply_deltas(self, text, distilled_output, current_gram, current_dict):
        """ Apply the deltas to the current grammar (Path) and dictionary (Path). """
        from beni.core.morphotactic.dabax import DabaX, get_dabax

        gram_delta = distilled_output.sebeni_gram
        dict_delta = distilled_output.sebeni_dict
        # Initial Morphology State
        dx = get_dabax(self.lang, gram=current_gram, ldict=current_dict, process=True)
        sent = dx.loader(text) 
        ms = MorphologyScorer()

        adjusted_gram = self.apply_gram_deltas(current_gram, gram_delta) if gram_delta else current_gram.read_text(encoding="utf-8")
        adjusted_dict = self.apply_dict_deltas(current_dict, dict_delta) if dict_delta else current_dict.read_text(encoding="utf-8")

        temp_gram = None
        temp_dict = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gram", mode='w', encoding='utf-8') as temp_gram_file:
                temp_gram_file.write(adjusted_gram)
                temp_gram = Path(temp_gram_file.name)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".dict", mode='w', encoding='utf-8') as temp_dict_file:
                temp_dict_file.write(adjusted_dict)
                temp_dict = Path(temp_dict_file.name)

            dx_prime = DabaX(self.lang, gram=temp_gram, ldict=temp_dict) # Derivative State of DabaX for Validation
            sent_prime = dx_prime.loader(text)

            state1 = round(ms.phi(sent[0]), 3)
            state2 = round(ms.phi(sent_prime[0]), 3)

            if(state1 > state2):
                logger.warning(f"Warning: Distillation resulted in lower morphological score: {state1} -> {state2}")
                return current_gram.read_text(encoding="utf-8"), current_dict.read_text(encoding="utf-8"), state1 > state2
            
            elif state1 == state2:
                logger.info(f"Distillation resulted in unchanged morphological score: {state1} -> {state2}")
                return current_gram.read_text(encoding="utf-8"), current_dict.read_text(encoding="utf-8"), state1 == state2

            else:
                print(f"Distillation improved morphological score: {state1} -> {state2}")
                logger.info(f"Distillation improved morphological score: {state1} -> {state2}")
                return adjusted_gram, adjusted_dict, state1 < state2

        except Exception as e:
            logger.error(f"Error with current pattern. Reverting back to old State {e}")
            return current_gram, current_dict

        finally:
            if temp_gram and temp_gram.exists():
                temp_gram.unlink()
            if temp_dict and temp_dict.exists():
                temp_dict.unlink()

    @staticmethod
    def _as_text(source: Union[str, Path]) -> str:
        if isinstance(source, Path):
            return source.read_text(encoding="utf-8") if source.exists() else ""
        return str(source or "")

    @staticmethod
    def _normalize_delta_tag(tag: str) -> str:
        """Map ``[REMOVE…]`` onto ``[DELETE…]`` so both flags share one path."""
        stripped = (tag or "").strip()
        if stripped == "[REMOVE]":
            return "[DELETE]"
        if stripped.startswith("[REMOVE:"):
            return "[DELETE:" + stripped[len("[REMOVE:") :]
        return stripped

    @staticmethod
    def _parse_delta_actions(delta_text: str) -> List[tuple]:
        if not (delta_text or "").strip():
            return []
        parts = re.split(
            r"(^\[(?:ADD|REPLACE|DELETE|REMOVE)[^\n]*\]$)",
            delta_text,
            flags=re.MULTILINE,
        )
        actions = []
        for i in range(1, len(parts), 2):
            tag = Distiller._normalize_delta_tag(parts[i])
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            actions.append((tag, content))
        return actions

    @staticmethod
    def _delete_target(tag: str, content: str) -> str:
        if tag.startswith("[DELETE:"):
            return tag.replace("[DELETE:", "", 1).rsplit("]", 1)[0].strip()
        return (content or "").strip()

    @staticmethod
    def _apply_one_gram_action(text: str, tag: str, content: str) -> str:
        try:
            if tag == "[DELETE]" or tag.startswith("[DELETE:"):
                target = Distiller._delete_target(tag, content)
                if not target:
                    return text
                pattern = Distiller._build_flexible_pattern(target)
                return re.sub(pattern, "", text, flags=re.MULTILINE)
            if tag.startswith("[REPLACE:"):
                target = tag.replace("[REPLACE:", "", 1).rsplit("]", 1)[0].strip()
                pattern = Distiller._build_flexible_pattern(target)
                if re.search(pattern, text, flags=re.MULTILINE):
                    return re.sub(pattern, lambda m: content, text, flags=re.MULTILINE)
                separator = "\n" if text else ""
                return f"{separator}{content}" if content else text
            if tag == "[ADD]":
                if not content:
                    return text
                separator = "\n" if text else ""
                return f"{text}{separator}{content}"
        except re.error as exc:
            logger.warning("Invalid grammar delta %s (%s); skipping", tag, exc)
            return text
        return text

    @staticmethod
    def _apply_one_dict_action(text: str, tag: str, content: str) -> str:
        try:
            if tag == "[DELETE]" or tag.startswith("[DELETE:"):
                target = Distiller._delete_target(tag, content)
                if not target:
                    return text
                escaped_target = re.escape(target)
                flexible_target = re.sub(r"\\\s|\s", r"\\s+", escaped_target)
                pattern = r"^" + flexible_target + r".*?(?=^\s*\\lx |\Z)"
                return re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)
            if tag.startswith("[REPLACE:"):
                target = tag.replace("[REPLACE:", "", 1).rsplit("]", 1)[0].strip()
                escaped_target = re.escape(target)
                flexible_target = re.sub(r"\\\s|\s", r"\\s+", escaped_target)
                pattern = r"^" + flexible_target + r".*?(?=^\s*\\lx |\Z)"
                if re.search(pattern, text, flags=re.DOTALL | re.MULTILINE):
                    return re.sub(
                        pattern, lambda m: content, text, flags=re.DOTALL | re.MULTILINE
                    )
                return f"{text}\n\n{content}" if content else text
            if tag == "[ADD]":
                if not content:
                    return text
                return f"{text}\n\n{content}"
        except re.error as exc:
            logger.warning("Invalid dictionary delta %s (%s); skipping", tag, exc)
            return text
        return text

    @staticmethod
    def _build_flexible_pattern(target: str) -> str:
        """
        Builds a regex pattern that matches the target at the start of a line,
        allowing for flexible whitespace to handle LLM space normalization.
        """
        escaped = re.escape(target)
        flexible = re.sub(r'\\\s|\s', r'\\s+', escaped)
        return r'^' + flexible + r'[^\r\n]*(?:\r?\n(?=\s+)[^\r\n]*)*'

    @staticmethod
    def apply_gram_deltas(
        current_gram: Union[str, Path],
        delta_text: str,
        keep_if: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """Apply ADD / REPLACE / DELETE / REMOVE tags to grammar text.

        ``keep_if`` is checked after each action. If it returns False, that
        tagged change is skipped and the previous grammar is kept.
        """
        current_gram = Distiller._as_text(current_gram)
        if not (delta_text or "").strip():
            return current_gram

        updated = current_gram
        applied = False
        for tag, content in Distiller._parse_delta_actions(delta_text):
            candidate = Distiller._apply_one_gram_action(updated, tag, content)
            if candidate == updated:
                continue
            if keep_if is not None and not keep_if(candidate):
                logger.warning(
                    "Skipping grammar %s; candidate gram/dict not parseable", tag
                )
                continue
            updated = candidate
            applied = True

        if not applied:
            return current_gram
        updated = re.sub(r"\n{3,}", "\n\n", updated).strip()
        return updated + "\n" if updated else ""

    @staticmethod
    def apply_dict_deltas(
        current_dict: Union[str, Path],
        delta_text: str,
        keep_if: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """Apply ADD / REPLACE / DELETE / REMOVE tags to dictionary text.

        ``keep_if`` is checked after each action. If it returns False, that
        tagged change is skipped and the previous dictionary is kept.
        """
        current_dict = Distiller._as_text(current_dict)
        if not (delta_text or "").strip():
            return current_dict

        updated = current_dict
        applied = False
        for tag, content in Distiller._parse_delta_actions(delta_text):
            candidate = Distiller._apply_one_dict_action(updated, tag, content)
            if candidate == updated:
                continue
            if keep_if is not None and not keep_if(candidate):
                logger.warning(
                    "Skipping dictionary %s; candidate gram/dict not parseable", tag
                )
                continue
            updated = candidate
            applied = True

        if not applied:
            return current_dict
        updated = re.sub(r"\n{3,}", "\n\n", updated).strip()
        return updated + "\n" if updated else ""
