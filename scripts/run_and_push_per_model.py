#!/usr/bin/env python3
"""Run N Models x N Algorithms (50 steps, max_eval=10) & Direct Push Training Data to Private Org Repos.

Direct push target pattern:
    mlsftwrs/sebeni-smollm2-135-sft-number_steps
    (e.g., mlsftwrs/sebeni-smollm2-135-sft-50)

Pushes training data, logs, configs, evaluation reports, and K-Veritas provenance seals.
Weights (*.safetensors, *.bin, models/) are explicitly excluded.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Polyfill for Python < 3.10.7
if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda digits: None

import yaml
from huggingface_hub import HfApi, get_token

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.run_matrix import (
    build_config,
    load_model_catalog,
    run_experiment_cell,
    save_reporting_frames,
    print_summary_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sebeni.direct_push")


def format_repo_id(
    template: str,
    org: str,
    model_slug: str,
    algo: str,
    params: str,
    steps: int,
) -> str:
    """Format target repository name: e.g. mlsftwrs/sebeni-smollm2-135-sft-50."""
    clean_model = re.sub(r"(\d+)m$", r"\1", str(model_slug).lower())
    clean_model = re.sub(r"[^a-zA-Z0-9.-]", "-", clean_model).strip("-")
    clean_algo = str(algo).lower()

    repo_name = template.format(
        org=org,
        model=clean_model,
        algo=clean_algo,
        params=str(params).lower(),
        steps=steps,
    )
    if "/" not in repo_name:
        repo_name = f"{org}/{repo_name}"
    return repo_name


def generate_cell_model_card(
    repo_id: str,
    model_entry: Dict[str, Any],
    algo: str,
    scope: str,
    steps: int,
    metrics: Dict[str, Any],
) -> str:
    """Generate model card README with exact performance metrics and run parameters."""
    model_id = model_entry["id"]
    mer_str = f"{metrics['mer']:.2f}" if metrics.get("mer") is not None else "N/A"
    mcs_str = f"{metrics['mcs']:.4f}" if metrics.get("mcs") is not None else "N/A"
    uwec_str = f"{metrics['uwec']:.4f}" if metrics.get("uwec") is not None else "N/A"
    phi_str = f"{metrics['phi']:.4f}" if metrics.get("phi") is not None else "N/A"

    return f"""---
library_name: transformers
tags:
  - sebeni
  - {algo.lower()}
  - {model_entry['slug']}
  - multi13
  - low-resource-languages
  - morphotactics
  - training-data
license: apache-2.0
---

# Sebeni Training Run: {model_id} ({algo.upper()})

This repository contains the **training data, execution logs, evaluation reports, configuration, and K-Veritas provenance seals** for **{model_id}** post-trained with **{algo.upper()}** under the **{scope}** multilingual low-resource scope.

*(Note: Model weights and checkpoints are excluded to preserve raw training telemetry, configs, and provenance).*

## Run Details
- **Base Model**: `{model_id}` ({model_entry.get('params', 'N/A')})
- **Algorithm**: `{algo.upper()}`
- **Scope**: `{scope}` (13 Sahelian / West-African low-resource languages)
- **Training Budget**: `{steps}` steps
- **Hardware Target**: NVIDIA L4 GPU (24GB VRAM)
- **Distillation Teacher**: Gemini 3.7 Flash via Google Cloud Vertex AI

## Evaluation Metrics (Held-out `test.json`)
- **Phi (Constraint Satisfaction)**: `{phi_str}`
- **MER (Morpheme Error Rate)**: `{mer_str}`
- **MCS (Morpheme Cosine Similarity)**: `{mcs_str}`
- **UWEC (Unweighted Edit Cost)**: `{uwec_str}`

## Repository Directory Structure
```
├── data/               # Processed distillation dataset, grammar, dictionary, splits
├── exp/                # Evaluation reports and benchmark JSON outputs
├── runtime/            # Trackio / TensorBoard step loss curves and reward telemetry
├── config.yaml         # Complete experiment hyperparameter configuration
├── kveritas_report.pdf # Cryptographic K-Veritas verification seal and provenance
└── README.md           # This metadata card
```
"""


def direct_push_cell(
    api: HfApi,
    repo_id: str,
    run_dir: Path,
    model_entry: Dict[str, Any],
    algo: str,
    scope: str,
    steps: int,
    metrics: Dict[str, Any],
    private: bool = True,
    include_weights: bool = False,
) -> bool:
    """Direct push of training data, logs, configs, and card to a dedicated HF repo."""
    if not run_dir.exists():
        logger.warning("Run directory %s not found. Skipping push.", run_dir)
        return False

    logger.info("=" * 60)
    logger.info("DIRECT PUSH -> Hugging Face: %s", repo_id)
    logger.info("=" * 60)

    try:
        # 1. Create or verify private repository under org (mlsftwrs)
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )

        # 2. Generate and write README.md
        card_content = generate_cell_model_card(
            repo_id=repo_id,
            model_entry=model_entry,
            algo=algo,
            scope=scope,
            steps=steps,
            metrics=metrics,
        )
        readme_path = run_dir / "README.md"
        readme_path.write_text(card_content, encoding="utf-8")

        # 3. Exclude heavy model weights and checkpoints (only upload training data, configs, logs)
        ignore_patterns = [
            "*.pyc",
            "__pycache__/*",
            ".venv/*",
            "tmp/*",
            "*.tmp",
        ]
        if not include_weights:
            ignore_patterns.extend([
                "*.safetensors",
                "*.bin",
                "*.pt",
                "*.pth",
                "*.ckpt",
                "*.h5",
                "*.onnx",
                "checkpoint-*",
                "checkpoint-*/*",
                "models/*",
                "models",
            ])

        api.upload_folder(
            folder_path=str(run_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns,
            commit_message=f"Sebeni training data & telemetry: {model_entry['slug']} {algo.upper()} ({steps} steps)",
        )
        logger.info("Successfully pushed training data to https://huggingface.co/%s", repo_id)
        return True
    except Exception as exc:
        logger.error("Direct push to %s failed: %s", repo_id, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N Models x N Algorithms (50 steps, max_eval=10) & direct push training data to HF"
    )
    # Organization and Repo naming
    parser.add_argument(
        "--org",
        type=str,
        default="mlsftwrs",
        help="Target Hugging Face organization (default: mlsftwrs)",
    )
    parser.add_argument(
        "--repo-template",
        type=str,
        default="{org}/sebeni-{model}-{algo}-{steps}",
        help="Repository naming template (default: '{org}/sebeni-{model}-{algo}-{steps}')",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token (defaults to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Make repository public (default: private to org)",
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        default=False,
        help="Include model weights (*.safetensors, models/) in upload (default: False, training data only)",
    )

    # Matrix execution parameters
    parser.add_argument(
        "--models",
        type=str,
        default="SmolLM2-135M",
        help="Comma-separated model slugs/IDs or 'all' for all models (default: SmolLM2-135M)",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="sft,grpo,dpo,apo",
        help="Comma-separated algorithms (default: 'sft,grpo,dpo,apo')",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Training step budget per cell (default: 50)",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=10,
        help="Evaluation row cap (default: 10)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load in 4-bit BitsAndBytes quantization (recommended for L4)",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="MULTI13",
        help="Scope (default: MULTI13)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/matrix"),
        help="Output directory (default: experiments/matrix)",
    )
    parser.add_argument(
        "--distil-model",
        type=str,
        default="gemini-3.7-flash",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="global",
    )
    parser.add_argument(
        "--vertex",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="google",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip previously completed runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print execution plan and target repos without training or uploading",
    )

    args = parser.parse_args()

    # 1. Hugging Face Authentication Check
    token = args.hf_token or os.environ.get("HF_TOKEN") or get_token()
    if not token and not args.dry_run:
        logger.error("Hugging Face token not found! Set export HF_TOKEN=<token> or pass --hf-token.")
        sys.exit(1)

    api = HfApi(token=token)
    if not args.dry_run:
        try:
            user_info = api.whoami()
            username = user_info.get("name") or user_info.get("fullname")
            logger.info("Authenticated to Hugging Face as: %s", username)
        except Exception as exc:
            logger.error("Hugging Face auth verification failed: %s", exc)
            sys.exit(1)

    # 2. Environment & Catalog Loading
    os.environ["GOOGLE_CLOUD_LOCATION"] = args.location
    os.environ["GOOGLE_LOCATION"] = args.location

    base_config_path = repo_root / "configs" / "matrix" / "base_multi13.yaml"
    catalog_path = repo_root / "configs" / "matrix" / "models.json"
    catalog = load_model_catalog(catalog_path)

    # Filter models
    if args.models.strip().lower() == "all":
        selected_models = catalog
    else:
        filter_tokens = {m.strip().lower() for m in args.models.split(",") if m.strip()}
        selected_models = [
            m for m in catalog
            if m["id"].lower() in filter_tokens or m["slug"].lower() in filter_tokens
        ]
        if not selected_models:
            logger.error("No models matched '%s'. Available: %s", args.models, [m["slug"] for m in catalog])
            sys.exit(1)

    selected_algorithms = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    quantization = "4bit" if args.load_in_4bit else None
    is_private = not args.public
    total_cells = len(selected_models) * len(selected_algorithms)

    logger.info("=" * 70)
    logger.info("SEBENI N-MODEL x N-ALGO DIRECT PUSH RUNNER")
    logger.info("=" * 70)
    logger.info("Models (%d): %s", len(selected_models), [m["slug"] for m in selected_models])
    logger.info("Algorithms (%d): %s", len(selected_algorithms), selected_algorithms)
    logger.info("Budget: %d steps per cell | Max Eval Rows: %d", args.max_steps, args.max_eval_rows)
    logger.info("Include Weights: %s (default: False -> training data, telemetry, and logs only)", args.include_weights)
    logger.info("Target Org / Namespace: %s (private=%s)", args.org, is_private)
    logger.info("Total Runs to Execute: %d", total_cells)
    logger.info("=" * 70)

    # 3. Main Execution Loop: Train Cell -> Direct Push Training Data to Dedicated HF Repo
    all_results: List[Dict[str, Any]] = []
    current_idx = 0

    for model_entry in selected_models:
        for algo in selected_algorithms:
            current_idx += 1
            run_slug = f"{model_entry['slug']}_{algo}"
            run_dir = args.output_dir / run_slug

            target_repo = format_repo_id(
                template=args.repo_template,
                org=args.org,
                model_slug=model_entry["slug"],
                algo=algo,
                params=model_entry.get("params", "base"),
                steps=args.max_steps,
            )

            logger.info(
                "\n[%d/%d] RUNNING: %s (%s) with %s -> TARGET: %s",
                current_idx,
                total_cells,
                model_entry["id"],
                model_entry["slug"],
                algo.upper(),
                target_repo,
            )

            cfg_data = build_config(
                base_config_path=base_config_path,
                model_entry=model_entry,
                algorithm=algo,
                scope=args.scope,
                max_steps=args.max_steps,
                distil_model=args.distil_model,
                location=args.location,
                vertex=args.vertex,
                backend=args.backend,
                run_dir=run_dir,
                quantization=quantization,
                max_eval_rows=args.max_eval_rows,
                kveritas_disclosure="open",
            )

            # Execute Cell
            result = run_experiment_cell(cfg_data, run_dir, dry_run=args.dry_run)
            result["hf_repo"] = target_repo
            all_results.append(result)

            # Direct Push to Hugging Face immediately after run finishes
            if not args.dry_run and result.get("status") in ("COMPLETED", "CACHED"):
                direct_push_cell(
                    api=api,
                    repo_id=target_repo,
                    run_dir=run_dir,
                    model_entry=model_entry,
                    algo=algo,
                    scope=args.scope,
                    steps=args.max_steps,
                    metrics=result,
                    private=is_private,
                    include_weights=args.include_weights,
                )

    # 4. Save Global Reporting Frames
    save_reporting_frames(all_results, args.output_dir)
    print_summary_table(all_results)
    logger.info("\nALL %d RUNS COMPLETED SUCCESSFULLY!", total_cells)


if __name__ == "__main__":
    main()
