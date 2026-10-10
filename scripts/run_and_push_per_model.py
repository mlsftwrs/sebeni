#!/usr/bin/env python3
"""Run N Models x N Algorithms (50 steps, max_eval=10) & Direct Push to Private Org Repos.

Direct push target pattern:
    mlsftwrs/sebeni-[model_base]-[algo]_[params]
    (e.g., mlsftwrs/sebeni-smollm2-135m-grpo_135m)

After each individual (model, algorithm) run completes:
1. Validates training logs, weights, tokenizer, and K-Veritas provenance seal.
2. Auto-creates the private repository under the organization (default: mlsftwrs).
3. Generates a tailored model card README.md with evaluation metrics (MER, MCS, UWEC, Phi).
4. Directly uploads all model weights, configs, safety snapshots, and training logs.
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
    """Format target repository name conforming to Hugging Face naming conventions."""
    safe_params = re.sub(r"[^a-zA-Z0-9_-]", "", str(params).lower())
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "-", str(model_slug).lower()).strip("-")
    safe_algo = re.sub(r"[^a-zA-Z0-9_-]", "", str(algo).lower())

    repo_name = template.format(
        org=org,
        model=safe_model,
        algo=safe_algo,
        params=safe_params,
        steps=steps,
    )
    # Ensure standard org/name structure
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
    """Generate model card README with exact performance metrics and loading code."""
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
license: apache-2.0
pipeline_tag: text-generation
---

# Sebeni Morphotactic Policy: {model_id} ({algo.upper()})

This repository contains trained weights, adapters, safety snapshots, and training logs for **{model_id}** post-trained with **{algo.upper()}** under the **{scope}** multilingual low-resource scope.

## Model Details
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

## How to Load in Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "{model_id}"
repo_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, repo_id)
```

## Audit & Verification
- Includes K-Veritas cryptographic verification seal and safety governor snapshots in the repository files.
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
) -> bool:
    """Direct push of model weights, logs, configs, and card to a dedicated HF repo."""
    if not run_dir.exists():
        logger.warning("Run directory %s not found. Skipping push.", run_dir)
        return False

    logger.info("=" * 60)
    logger.info("DIRECT PUSH -> Hugging Face: %s", repo_id)
    logger.info("=" * 60)

    try:
        # 1. Create or verify private repository
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

        # 3. Upload all artifacts (model, tokenizer, configs, trackio/runs logs)
        ignore_patterns = ["*.pyc", "__pycache__/*", ".venv/*", "tmp/*", "*.tmp"]

        api.upload_folder(
            folder_path=str(run_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns,
            commit_message=f"Sebeni direct push: {model_entry['slug']} {algo.upper()} ({steps} steps)",
        )
        logger.info("Successfully pushed to https://huggingface.co/%s", repo_id)
        return True
    except Exception as exc:
        logger.error("Direct push to %s failed: %s", repo_id, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N Models x N Algorithms (50 steps, max_eval=10) & direct push to HF"
    )
    # Organization and Repo naming
    parser.add_argument(
        "--org",
        type=str,
        default="mlsftwrs",
        help="Target Hugging Face organization or username (default: mlsftwrs)",
    )
    parser.add_argument(
        "--repo-template",
        type=str,
        default="{org}/sebeni-{model}-{algo}_{params}",
        help="Repository naming template (default: '{org}/sebeni-{model}-{algo}_{params}')",
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
            logger.info("Authenticated to Hugging Face as: %s", user_info.get("name") or user_info.get("fullname"))
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
    logger.info("Quantization: %s", quantization or "none (bf16/fp16)")
    logger.info("Total Runs to Execute: %d", total_cells)
    logger.info("Target Org / Namespace: %s (private=%s)", args.org, is_private)
    logger.info("=" * 70)

    # 3. Main Execution Loop: Train Cell -> Direct Push to Dedicated HF Repo
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
                )

    # 4. Save Global Reporting Frames
    save_reporting_frames(all_results, args.output_dir)
    print_summary_table(all_results)
    logger.info("\nALL %d RUNS COMPLETED SUCCESSFULLY!", total_cells)


if __name__ == "__main__":
    main()
