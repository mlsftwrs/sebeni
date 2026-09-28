"""Canonical Sebeni pipeline: distill once, train one arm, evaluate held-out data.

``init``, ``distill``, ``train``, ``eval``, and ``exp`` all call this module.
Policy updates do not promote grammar or dictionary checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from beni.core.language import resolve_scope
from beni.core.srl.config import MasterConfig
from beni.utils import config as cfg

logger = logging.getLogger(__name__)


def frozen_marker(mc: MasterConfig) -> Path:
    root = Path(mc.working_dir or cfg.get_workdir().root)
    return root / "data" / "resources" / "frozen.json"


def resources_frozen(mc: MasterConfig) -> bool:
    return bool(mc.experiment.freeze_resources) and frozen_marker(mc).is_file()


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _scope(mc: MasterConfig):
    if mc.data.languages:
        return resolve_scope(mc.data.languages)
    if mc.experiment.scope:
        return resolve_scope(mc.experiment.scope)
    return resolve_scope(mc.data.default_lang or "bam")


def load_train_records(mc: MasterConfig) -> Dict[str, Any]:
    """Train rows for distill and arms. Never reads ``test.json``."""
    from beni.data.datasets import assert_train_source, load_experiment_records
    from beni.data.datasets import SebeniDataLoader

    scope = _scope(mc)
    mc.data.languages = list(scope.languages)
    mc.experiment.scope = scope.label
    if mc.data.source:
        source = mc.data.source
        paths = source if isinstance(source, list) else [source]
        for item in paths:
            assert_train_source(item)
        records = SebeniDataLoader(mc.data).load(source)
        return {
            "records": records,
            "counts": _counts(records),
            "dropped_bbo": len(scope.dropped),
            "source": str(source),
            "scope": scope,
        }
    if str(mc.experiment.dataset or "raw").lower() == "packaged":
        loaded = load_experiment_records(scope.languages)
        loaded["scope"] = scope
        loaded["request_dropped"] = list(scope.dropped)
        return loaded
    records = SebeniDataLoader(mc.data).load_sebeni(split="train")
    return {
        "records": records,
        "counts": _counts(records),
        "dropped_bbo": len(scope.dropped),
        "source": str(cfg.DATA_DIR / "raw"),
        "scope": scope,
    }


def load_eval_records(mc: MasterConfig) -> List[Dict[str, str]]:
    """Held-out paragraphs from packaged ``test.json``, filtered to the run scope."""
    from beni.data.datasets import SebeniDataLoader

    scope = _scope(mc)
    previous = mc.data.languages
    mc.data.languages = list(scope.languages)
    try:
        return SebeniDataLoader(mc.data).load_sebeni(split="test")
    finally:
        mc.data.languages = previous


def _counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in records or []:
        lang = str(row.get("lang") or row.get("language") or "")
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return counts


def _distiller_for(mc: MasterConfig, group: str):
    from beni.core.morphotactic.distil.distillation import Distiller

    distiller = Distiller(
        lang_code=group,
        backend=mc.distillation.selected_backend,
        model=mc.distillation.model,
        working_dir=mc.distillation.working_dir or mc.working_dir,
        vertex=mc.distillation.vertex,
        base_url=mc.distillation.base_url,
        gguf_path=mc.distillation.gguf_path,
        n_ctx=mc.distillation.n_ctx,
        max_input_chars=mc.distillation.max_input_chars,
    )
    distiller.handle_baselines()
    return distiller


def _snapshot_language(distiller, lang: str, root: Path) -> Dict[str, Any]:
    dest = root / "data" / "resources" / lang
    dest.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(distiller.baselines_dir)
    copied = {}
    for name, target in (
        ("baseline.gram", "baseline.gram"),
        ("baseline.dict", "baseline.dict"),
    ):
        src = baseline_dir / name
        if src.is_file():
            shutil.copy2(src, dest / target)
            copied[target] = _sha256(dest / target)
    gram = Path(distiller.gram_path) if distiller.gram_path else None
    ldict = Path(distiller.dict_path) if distiller.dict_path else None
    if gram and gram.is_file():
        shutil.copy2(gram, dest / "distilled.gram")
        copied["distilled.gram"] = _sha256(dest / "distilled.gram")
    if ldict and ldict.is_file():
        shutil.copy2(ldict, dest / "distilled.dict")
        copied["distilled.dict"] = _sha256(dest / "distilled.dict")
    payload = {
        "language": lang,
        "checkpoint_id": distiller.checkpoint_id(),
        "hashes": copied,
        "directory": str(dest),
    }
    (dest / "hashes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_manifest(mc: MasterConfig, **fields: Any) -> Path:
    root = Path(mc.working_dir or cfg.get_workdir().root)
    path = root / "exp" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    current: Dict[str, Any] = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    scope = _scope(mc)
    current.update(
        {
            "model": mc.model.model_name,
            "scope": scope.label,
            "languages": list(scope.languages),
            "method": mc.algorithm,
            "seed": getattr(_active_trainer(mc), "seed", None),
            "config": str(getattr(mc, "_config_file_dir", "") or ""),
            "commit": _git_commit(),
            "freeze_resources": True,
        }
    )
    current.update(fields)
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _active_trainer(mc: MasterConfig):
    if mc.algorithm == "dpo":
        return mc.dpo
    if mc.algorithm == "apo":
        return mc.apo
    if mc.algorithm == "sft":
        return mc.sft
    return mc.trainer


def run_distill(mc: MasterConfig, records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """SAMPG once per language, then freeze G and D."""
    from beni.core.safety.governor import SafetyGovernor
    from beni.core.srl.algorithm1 import distill_language, group_texts_by_language, records_text_langs

    if mc.working_dir:
        cfg.set_working_dir(mc.working_dir)
    loaded = None
    if records is None:
        loaded = load_train_records(mc)
        records = loaded["records"]
    scope = _scope(mc)
    default_lang = mc.data.default_lang or (scope.languages[0] if scope.languages else "bam")
    grouped = group_texts_by_language(records_text_langs(records or [], default_lang), default_lang=default_lang)
    allowed = set(scope.languages)
    governor = SafetyGovernor(mc.safety.to_spec(tau=mc.distillation.tau, kl_beta=mc.trainer.beta))
    decisions = {}
    resources = {}
    root = Path(mc.working_dir or cfg.get_workdir().root)
    if not mc.distillation.enabled:
        decisions["disabled"] = "distillation disabled"
    else:
        for group, texts in grouped.items():
            if group not in allowed:
                continue
            try:
                distiller = _distiller_for(mc, group)
                decision = distill_language(
                    texts,
                    distiller,
                    governor,
                    mc.distillation.tau,
                    hitl=mc.distillation.hitl,
                )
                decisions[group] = {
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    "phi": decision.phi,
                    "phi_prime": decision.phi_prime,
                }
                resources[group] = _snapshot_language(distiller, group, root)
            except Exception as exc:
                logger.warning(
                    "Distill %s failed; keeping previous G, D (%s)", group, exc
                )
                decisions[group] = {
                    "allowed": False,
                    "reason": "distill_error",
                    "phi": 0.0,
                    "phi_prime": 0.0,
                    "error": str(exc),
                }
    mc.experiment.freeze_resources = True
    marker = {
        "frozen": True,
        "scope": scope.label,
        "languages": list(scope.languages),
        "dropped": list(scope.dropped),
        "decisions": decisions,
        "resources": resources,
    }
    frozen_marker(mc).parent.mkdir(parents=True, exist_ok=True)
    frozen_marker(mc).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    fingerprint = hashlib.sha256(
        json.dumps(records or [], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_manifest(
        mc,
        stage="distill",
        dataset_fingerprint=fingerprint,
        language_counts=(loaded or {}).get("counts") or _counts(records or []),
        dropped_bbo=(loaded or {}).get("dropped_bbo", len(scope.dropped)),
        train_source=(loaded or {}).get("source"),
        resources=resources,
        distill=decisions,
    )
    return marker


def run_arm(mc: MasterConfig, records: Optional[List[Dict[str, Any]]] = None):
    """Train exactly one arm against frozen resources."""
    if mc.working_dir:
        cfg.set_working_dir(mc.working_dir)
    loaded = None
    if records is None:
        loaded = load_train_records(mc)
        records = loaded["records"]
    if not resources_frozen(mc):
        run_distill(mc, records)
    if str(mc.trainer.framework).lower() == "jax" and mc.algorithm == "sft":
        raise ValueError("SFT is torch-only. Set trainer.framework: torch for algorithm: sft.")
    from beni.core.srl.unified import SRLTrainer

    trainer = SRLTrainer(mc)
    trainer.train(records)
    _write_manifest(
        mc,
        stage="train",
        language_counts=(loaded or {}).get("counts") or _counts(records or []),
        train_source=(loaded or {}).get("source"),
    )
    return trainer


def run_eval(
    mc: MasterConfig,
    records: Optional[List[Dict[str, Any]]] = None,
    generate: Optional[Callable[[str, str], str]] = None,
    policy: Any = None,
) -> Dict[str, Any]:
    """Score held-out test text. MER, MCS, and UWEC pool by morpheme or token count."""
    from beni.core.compute.helpers import mer_edit_counts, mcs_mismatch_counts
    from beni.core.compute.metrics import MorphologyScorer
    from beni.core.compute.uwec import array_module, uwec
    from beni.core.morphotactic.dabax import get_dabax
    from beni.core.safety.governor import SafetyGovernor

    try:
        import jax.numpy as jnp  # type: ignore

        xp = array_module(jnp)
    except Exception:
        import numpy as np

        xp = array_module(np)

    if mc.working_dir:
        cfg.set_working_dir(mc.working_dir)
    if records is None:
        records = load_eval_records(mc)
    scope = _scope(mc)
    grouped: Dict[str, List[str]] = {}
    for row in records or []:
        lang = str(row.get("lang") or row.get("language") or "")
        text = str(row.get("text") or "")
        if not text or (scope.languages and lang not in scope.languages):
            continue
        grouped.setdefault(lang, []).append(text)

    scorer = MorphologyScorer()
    by_language = {}
    cap = max(int(mc.experiment.max_eval_rows or 0), 0)
    scope_mer_ops = 0
    scope_mer_n = 0
    scope_mcs_mis = 0.0
    scope_mcs_n = 0.0
    scope_uwec_sum = 0.0
    scope_uwec_n = 0
    decode = generate
    if decode is None and policy is not None:

        def _from_policy(text: str, _lang: str) -> str:
            return policy.generate(text)

        decode = _from_policy

    def _morphemes(sentence):
        return [form for token in sentence.tokens for form in token.best_morphemes()]

    def _stages(sentence):
        return [token.stage for token in sentence.tokens]

    for group, texts in grouped.items():
        distiller = _distiller_for(mc, group)
        dabax = get_dabax(
            group,
            gram=distiller.gram_path,
            ldict=distiller.dict_path,
            process=True,
            runtime_dir=cfg.get_workdir().runtime,
        )
        sentences = []
        for text in texts:
            try:
                sentences.extend(dabax.loader(text) or [])
            except Exception:
                continue
        phi = scorer.phi_corpus(sentences)["avg"] if sentences else 0.0
        mer_ops = 0
        mer_n = 0
        mcs_mis = 0.0
        mcs_n = 0.0
        uwec_sum = 0.0
        uwec_n = 0
        n_scored = 0
        if decode is not None and cap:
            for text in texts[:cap]:
                try:
                    completion = decode(text, group)
                except Exception:
                    continue
                if not completion:
                    continue
                try:
                    hyp_sentences = dabax.loader(completion) or []
                except Exception:
                    hyp_sentences = []
                try:
                    ref_sentences = dabax.loader(text) or []
                except Exception:
                    ref_sentences = []
                if not ref_sentences or not hyp_sentences:
                    continue
                ref_sent, hyp_sent = ref_sentences[0], hyp_sentences[0]
                ops, n_ref = mer_edit_counts(_morphemes(ref_sent), _morphemes(hyp_sent), xp=xp)
                mer_ops += int(ops)
                mer_n += int(n_ref)
                mis, n_tok = mcs_mismatch_counts(_stages(hyp_sent), _stages(ref_sent), xp=xp)
                mcs_mis += float(mis)
                mcs_n += float(n_tok)
                pi_theta = pi_ref = None
                if policy is not None and hasattr(policy, "token_probabilities"):
                    try:
                        pi_theta, pi_ref = policy.token_probabilities(completion)
                    except Exception:
                        pi_theta, pi_ref = None, None
                hyp_stages = _stages(hyp_sent)
                _, mean = uwec(
                    hyp_stages,
                    pi_theta,
                    pi_ref,
                    beta=scorer.beta,
                    eps=scorer.eps,
                    xp=xp,
                )
                n_u = len(hyp_stages)
                uwec_sum += float(mean) * n_u
                uwec_n += n_u
                n_scored += 1

        mer = (mer_ops / mer_n) if mer_n else None
        mcs = (mcs_mis / mcs_n) if mcs_n else None
        uwec_val = (uwec_sum / uwec_n) if uwec_n else None
        scope_mer_ops += mer_ops
        scope_mer_n += mer_n
        scope_mcs_mis += mcs_mis
        scope_mcs_n += mcs_n
        scope_uwec_sum += uwec_sum
        scope_uwec_n += uwec_n
        by_language[group] = {
            "phi": phi,
            "mer": mer,
            "mcs": mcs,
            "uwec": uwec_val,
            "n_sentences": len(sentences),
            "n_scored": n_scored,
            "n_morphemes": mer_n,
            "n_tokens": int(mcs_n) if mcs_n else uwec_n,
            "checkpoint_id": distiller.checkpoint_id(),
            "language": group,
            "scope": scope.label,
        }
    ref_id = mc.model.ref_model_name or mc.model.model_name
    report = {
        "model": mc.model.model_name,
        "algorithm": mc.algorithm,
        "scope": scope.label,
        "languages": list(grouped.keys()) or list(scope.languages),
        "by_language": by_language,
        "phi": _macro(by_language, "phi"),
        "mer": (scope_mer_ops / scope_mer_n) if scope_mer_n else None,
        "mcs": (scope_mcs_mis / scope_mcs_n) if scope_mcs_n else None,
        "uwec": (scope_uwec_sum / scope_uwec_n) if scope_uwec_n else None,
        "n_morphemes": scope_mer_n,
        "n_tokens": int(scope_mcs_n) if scope_mcs_n else scope_uwec_n,
        "beta": scorer.beta,
        "eps": scorer.eps,
        "reference_model": ref_id,
        "tau": mc.distillation.tau,
    }
    out = cfg.get_workdir().exp / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    gov = SafetyGovernor(mc.safety.to_spec(tau=mc.distillation.tau, kl_beta=mc.trainer.beta))
    gov.record_snapshot(
        phi=report["phi"] or 0.0,
        tau=mc.distillation.tau,
        mer=report["mer"],
        mcs=report["mcs"],
        checkpoint_id=",".join(str(v.get("checkpoint_id") or "") for v in by_language.values()),
        algorithm=mc.algorithm,
        language=",".join(report["languages"]),
        extra={"scope": scope.label, "by_language": by_language, "uwec": report["uwec"]},
    )
    gov.write_snapshot(cfg.get_workdir().models)
    _write_manifest(mc, stage="eval", eval=str(out), scope=scope.label)
    return report


def _macro(by_language: Dict[str, Dict[str, Any]], key: str) -> Optional[float]:
    vals = [row[key] for row in by_language.values() if row.get(key) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def run_experiment(mc: MasterConfig) -> Dict[str, Any]:
    """Distill, train one arm, then evaluate. The experiment train set is the packaged jsonl."""
    mc.experiment.dataset = "packaged"
    mc.data.source = None
    loaded = load_train_records(mc)
    if not loaded["records"]:
        raise RuntimeError("Experiment train split is empty (dataset_300_samples.jsonl).")
    run_distill(mc, loaded["records"])
    trainer = run_arm(mc, loaded["records"])

    def _generate(text: str, _lang: str) -> str:
        return trainer.generate(text)

    return run_eval(mc, generate=_generate, policy=getattr(trainer, "plugin", trainer))
