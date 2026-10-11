#!/usr/bin/env python3
"""Re-evaluate Aligned Model Weights & Update Hugging Face Model Cards.

This script:
1. Loads post-aligned model weights (from local run directories or Hugging Face repos).
2. Runs post-training evaluation on held-out `beni/data/test.json`.
3. Computes genuine hypothesis-based Phi (Constraint Satisfaction), MER, MCS, and UWEC.
4. Regenerates canonical Section 10 model card tables and README.md.
5. Updates `README.md` and `eval_results.json` directly on Hugging Face Hub without re-training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Polyfill for Python < 3.10.7
if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda digits: None

try:
    from huggingface_hub import HfApi, get_token
except ImportError:
    HfApi = None
    get_token = lambda: None

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sebeni.recompute_eval")


def recompute_cell_eval(
    mc: Any,
    weights_source: str,
    run_dir: Path,
    max_eval_rows: int = 10,
) -> Dict[str, Any]:
    """Load post-trained weights and evaluate on held-out test.json."""
    from beni.core.pipeline import run_eval
    from beni.core.srl.config import SRLGrpoPrompt
    from beni.core.srl.unified import SRLTrainer

    logger.info("Initializing SRLTrainer for post-alignment evaluation...")
    trainer = SRLTrainer(mc)
    trainer.load_models()

    logger.info("Loading aligned weights from: %s", weights_source)
    trainer.load_aligned_weights(weights_source)
    if hasattr(trainer, "model") and trainer.model is not None:
        trainer.model.eval()

    tokenizer = getattr(trainer, "tokenizer", None)

    def _generate(text: str, lang: str) -> str:
        system_prompt = SRLGrpoPrompt(languages=[lang] if lang else None).prompt()
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            try:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return trainer.generate(formatted)
            except Exception:
                pass
        prompt = f"{system_prompt}\n\nUser: {text}\nAssistant:"
        return trainer.generate(prompt)

    # Set evaluation cap
    mc.experiment.max_eval_rows = max_eval_rows

    logger.info("Running evaluation on held-out test.json...")
    eval_report = run_eval(
        mc,
        generate=_generate,
        policy=getattr(trainer, "plugin", trainer),
    )
    return eval_report


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate aligned model weights and update Hugging Face model cards."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Specific Hugging Face repo ID (e.g. mlsftwrs/sebeni-smollm2-135-grpo-100)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="SmolLM2-135M",
        help="Comma-separated model identifiers or 'all'",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="grpo,sft",
        help="Comma-separated algorithms: sft, grpo, dpo, apo",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Step budget suffix used in repo naming (default: 100)",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="MULTI13",
        help="Language scope (default: MULTI13)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "experiments" / "matrix",
        help="Local experiment directory root",
    )
    parser.add_argument(
        "--hf-org",
        type=str,
        default="mlsftwrs",
        help="Hugging Face organization (default: mlsftwrs)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=10,
        help="Sentences to evaluate per language (default: 10)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=True,
        help="Push updated README.md and eval_results.json to HF Hub",
    )
    parser.add_argument(
        "--no-push",
        dest="push_to_hub",
        action="store_false",
        help="Do not push updates to HF Hub (local evaluation only)",
    )

    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token) if token else None
    if args.push_to_hub and not api:
        logger.error("Hugging Face token not found! Set HF_TOKEN or pass --hf-token.")
        sys.exit(1)

    from scripts.run_matrix import build_config, load_model_catalog
    from scripts.run_and_push_per_model import generate_cell_model_card, get_target_repo_id
    from beni.core.srl.config import MasterConfig
    from beni.utils import config as cfg

    catalog_path = repo_root / "configs" / "matrix" / "models.json"
    catalog = load_model_catalog(catalog_path)
    base_config_path = repo_root / "configs" / "matrix" / "base_multi13.yaml"

    selected_algos = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]

    # Collect cells to evaluate
    eval_cells = []
    if args.repo_id:
        # One or more explicit repo IDs (comma-separated)
        repos = [r.strip() for r in args.repo_id.split(",") if r.strip()]
        for repo in repos:
            matched_model = None
            matched_algo = "grpo"
            for m in catalog:
                if m["slug"] in repo.lower():
                    matched_model = m
                    break
            if not matched_model:
                matched_model = catalog[0]
            for a in ("grpo", "sft", "dpo", "apo"):
                if f"-{a}-" in repo.lower() or repo.lower().endswith(f"-{a}"):
                    matched_algo = a
                    break
            eval_cells.append({
                "repo_id": repo,
                "model_entry": matched_model,
                "algo": matched_algo,
                "steps": args.steps,
            })
    else:
        # Multi-model / algorithm matrix
        for m in catalog:
            if args.models.lower() != "all" and m["id"].lower() not in args.models.lower() and m["slug"].lower() not in args.models.lower():
                continue
            for algo in selected_algos:
                repo_id = get_target_repo_id(args.hf_org, m, algo, args.steps)
                eval_cells.append({
                    "repo_id": repo_id,
                    "model_entry": m,
                    "algo": algo,
                    "steps": args.steps,
                })

    logger.info("Found %d model cell(s) to re-evaluate.", len(eval_cells))

    for idx, cell in enumerate(eval_cells, 1):
        repo_id = cell["repo_id"]
        model_entry = cell["model_entry"]
        algo = cell["algo"]
        steps = cell["steps"]
        run_slug = f"{model_entry['slug']}_{algo}"
        run_dir = args.output_dir / run_slug

        logger.info("\n" + "=" * 60)
        logger.info("[%d/%d] RE-EVALUATING: %s (%s)", idx, len(eval_cells), repo_id, algo.upper())
        logger.info("=" * 60)

        # Build MasterConfig
        cfg_data = build_config(
            base_config_path=base_config_path,
            model_entry=model_entry,
            algorithm=algo,
            scope=args.scope,
            max_steps=steps,
            run_dir=run_dir,
            max_eval_rows=args.max_eval_rows,
        )
        mc = MasterConfig.from_dict(cfg_data)
        mc.apply_workdir(run_dir)

        # Locate aligned weights: local run_dir/models or HF repo
        local_models = run_dir / "models"
        has_local = local_models.exists() and any(local_models.iterdir())
        has_remote = False
        if not has_local and api is not None:
            try:
                has_remote = api.repo_exists(repo_id=repo_id, repo_type="model")
            except Exception:
                has_remote = False

        if not has_local and not has_remote:
            logger.warning(
                "Skipping %s [%s]: No aligned weights found locally in %s or on HF repo %s",
                model_entry["slug"],
                algo,
                local_models,
                repo_id,
            )
            continue

        weights_source = str(local_models) if has_local else repo_id

        try:
            eval_report = recompute_cell_eval(
                mc,
                weights_source=weights_source,
                run_dir=run_dir,
                max_eval_rows=args.max_eval_rows,
            )

            metrics = {
                "phi": eval_report.get("phi"),
                "mer": eval_report.get("mer"),
                "mcs": eval_report.get("mcs"),
                "uwec": eval_report.get("uwec"),
                "eval_report": eval_report,
                "status": "COMPLETED",
            }

            logger.info(
                "Recomputed Scores -> Phi: %.4f | MER: %.2f | MCS: %.4f | UWEC: %.4f",
                metrics["phi"] or 0.0,
                metrics["mer"] or 0.0,
                metrics["mcs"] or 0.0,
                metrics["uwec"] or 0.0,
            )

            # Save local eval results
            exp_dir = run_dir / "exp"
            exp_dir.mkdir(parents=True, exist_ok=True)
            eval_path = exp_dir / "eval.json"
            eval_path.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False), encoding="utf-8")
            shutil.copy2(eval_path, run_dir / "eval_results.json")

            # Regenerate model card
            card_content = generate_cell_model_card(
                repo_id=repo_id,
                model_entry=model_entry,
                algo=algo,
                scope=args.scope,
                steps=steps,
                metrics=metrics,
            )
            readme_path = run_dir / "README.md"
            readme_path.write_text(card_content, encoding="utf-8")
            logger.info("Saved updated model card to %s", readme_path)

            # Upload updated README.md and eval_results.json to HF Hub
            if args.push_to_hub and api:
                logger.info("Pushing updated model card & benchmark results to Hugging Face: %s...", repo_id)
                api.upload_file(
                    path_or_fileobj=str(readme_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="docs: update benchmark evaluation scores with post-aligned weights",
                )
                api.upload_file(
                    path_or_fileobj=str(eval_path),
                    path_in_repo="eval_results.json",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="docs: update held-out eval_results.json",
                )
                logger.info("Successfully updated Hugging Face repository: %s", repo_id)

        except Exception as exc:
            logger.error("Failed to recompute evaluation for %s: %s", repo_id, exc, exc_info=True)
            if args.push_to_hub and api:
                from scripts.run_and_push_per_model import push_error_to_hub
                import traceback
                push_error_to_hub(
                    api=api,
                    repo_id=repo_id,
                    run_dir=run_dir,
                    model_entry=model_entry,
                    algo=algo,
                    scope=args.scope,
                    steps=steps,
                    error_msg=str(exc),
                    traceback_str=traceback.format_exc(),
                    private=True,
                )


if __name__ == "__main__":
    main()
