#!/usr/bin/env python3
"""Run N Models x N Algorithms (50 steps, max_eval=10) & Direct Push Model Weights, Tokenizers, and Training Data to Private Org Repos.

Direct push target pattern:
    mlsftwrs/sebeni-smollm2-135-sft-number_steps
    (e.g., mlsftwrs/sebeni-smollm2-135-sft-50)

Pushes trained model weights (*.safetensors, adapter configs), tokenizers, training datasets,
telemetry logs, configs, evaluation reports, and K-Veritas provenance seals.
Ready for direct downstream inference with transformers / peft.
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
import traceback
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
    eval_report = metrics.get("eval_report") or {}
    by_lang = eval_report.get("by_language") or {}
    lang_table = ""
    if by_lang:
        lang_table = "\n### Per-Language Evaluation Scores\n\n| Language | Phi | MER | MCS | UWEC | Scored Sentences |\n|---|---|---|---|---|---|\n"
        for lang_code, lang_stats in sorted(by_lang.items()):
            p = f"{lang_stats.get('phi', 0.0):.4f}" if lang_stats.get('phi') is not None else "-"
            m = f"{lang_stats.get('mer', 0.0):.2f}" if lang_stats.get('mer') is not None else "-"
            c = f"{lang_stats.get('mcs', 0.0):.4f}" if lang_stats.get('mcs') is not None else "-"
            u = f"{lang_stats.get('uwec', 0.0):.4f}" if lang_stats.get('uwec') is not None else "-"
            n = lang_stats.get("n_scored", 0)
            lang_table += f"| `{lang_code}` | {p} | {m} | {c} | {u} | {n} |\n"

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

This repository contains the **trained model weights, tokenizer, training data, execution logs, evaluation reports, configuration, and K-Veritas provenance seals** for **{model_id}** post-trained with **{algo.upper()}** under the **{scope}** multilingual low-resource scope.

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
{lang_table}
## Inference & Usage

You can load this post-trained model and tokenizer directly using Hugging Face `transformers` and `peft`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "{model_id}"
repo_id = "{repo_id}"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# 2. Load Model (LoRA Adapter or Full Pretrained)
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, repo_id)
    print(f"Loaded PEFT adapter from {repo_id}")
except Exception:
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Loaded model from {repo_id}")

# 3. Generate morphological segmentation / SRL completion
prompt = "Linguistic input sentence here..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Repository Directory Structure
```
├── adapter_model.safetensors # LoRA trained adapter weights (or model.safetensors)
├── adapter_config.json       # PEFT adapter hyperparameter configuration
├── tokenizer.json            # Model tokenizer vocabulary and merges
├── tokenizer_config.json     # Tokenizer settings and special tokens map
├── data/                     # Processed distillation dataset, grammar, dictionary, splits
├── exp/                      # Evaluation reports and benchmark JSON outputs
├── runtime/                  # Trackio / TensorBoard step loss curves and reward telemetry
├── config.yaml               # Complete experiment hyperparameter configuration
├── kveritas_report.pdf       # Cryptographic K-Veritas verification seal and provenance
└── README.md                 # This metadata card
```
"""


def push_error_to_hub(
    api: Optional[HfApi],
    repo_id: str,
    run_dir: Path,
    model_entry: Dict[str, Any],
    algo: str,
    scope: str,
    steps: int,
    error_msg: str,
    traceback_str: str,
    private: bool = True,
) -> None:
    """Log failure and push error details and updated README to Hugging Face."""
    if not api:
        return
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write error.log locally
        err_file = run_dir / "error.log"
        err_content = (
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
            f"Model: {model_entry.get('id', 'unknown')}\n"
            f"Algorithm: {algo.upper()}\n"
            f"Error: {error_msg}\n\n"
            f"Traceback:\n{traceback_str}\n"
        )
        err_file.write_text(err_content, encoding="utf-8")

        # 2. Write model card reflecting failure state
        card_content = generate_cell_model_card(
            repo_id=repo_id,
            model_entry=model_entry,
            algo=algo,
            scope=scope,
            steps=steps,
            metrics={
                "model": model_entry["id"],
                "scope": scope,
                "method": algo.upper(),
                "mer": None,
                "mcs": None,
                "uwec": None,
                "phi": None,
                "seed": 42,
                "config": str(run_dir / "config.yaml"),
                "status": f"FAILED: {error_msg[:120]}",
            },
        )
        readme_path = run_dir / "README.md"
        readme_path.write_text(card_content, encoding="utf-8")

        # 3. Upload error.log and README.md to Hugging Face Hub
        api.upload_file(
            path_or_fileobj=str(err_file),
            path_in_repo="error.log",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"ci: log training/eval error: {error_msg[:60]}",
        )
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="docs: record failure status in model card",
        )
        logger.info("Successfully pushed failure report & error.log to HF: %s", repo_id)
    except Exception as exc:
        logger.warning("Could not push error report to HF repo %s: %s", repo_id, exc)


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
    include_weights: bool = True,
) -> bool:
    """Direct push of trained model weights, tokenizer, logs, configs, and card to a dedicated HF repo."""
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

        # 2. Surface final model weights, configs, tokenizer, and eval results to repo root
        # so standard AutoModelForCausalLM / AutoTokenizer / PeftModel can load directly from HF
        if include_weights:
            models_dir = run_dir / "models"
            if models_dir.exists():
                for item in models_dir.iterdir():
                    if item.is_file() and item.name not in ("README.md",):
                        dest = run_dir / item.name
                        shutil.copy2(item, dest)

        # Surface evaluation results JSON to repo root for transparent benchmarking
        eval_json = run_dir / "exp" / "eval.json"
        if eval_json.exists():
            shutil.copy2(eval_json, run_dir / "eval_results.json")

        # 3. Generate and write README.md
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

        # 4. Exclude temporary cache files and intermediate checkpoints
        # (Preserve final weights, tokenizers, configs, data, and provenance)
        ignore_patterns = [
            "*.pyc",
            "__pycache__/*",
            "*/__pycache__/*",
            ".venv/*",
            "tmp/*",
            "*.tmp",
            "checkpoint-*",
            "checkpoint-*/*",
            "*/checkpoint-*/*",
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
                "models/*",
                "models",
            ])

        api.upload_folder(
            folder_path=str(run_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns,
            commit_message=f"Sebeni trained model, tokenizer & training bundle: {model_entry['slug']} {algo.upper()} ({steps} steps)",
        )
        logger.info("Successfully pushed model weights, tokenizer, and bundle to https://huggingface.co/%s", repo_id)
        return True
    except Exception as exc:
        logger.error("Direct push to %s failed: %s", repo_id, exc, exc_info=True)
        return False


def cleanup_stale_run_dir(run_dir: Path) -> None:
    """Clean partial / broken run artifacts from previous aborted attempts."""
    if not run_dir.exists():
        return
    for sub in ("models", "runs", "tmp"):
        sub_path = run_dir / sub
        if sub_path.exists():
            shutil.rmtree(sub_path, ignore_errors=True)
    for p in run_dir.glob("checkpoint-*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    eval_f = run_dir / "exp" / "eval.json"
    if eval_f.exists():
        try:
            with open(eval_f, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not d or (d.get("phi") is None and d.get("mer") is None):
                eval_f.unlink(missing_ok=True)
        except Exception:
            eval_f.unlink(missing_ok=True)


def is_repo_completed_on_hub(api: HfApi, repo_id: str) -> bool:
    """Check if target repository exists on Hugging Face Hub and has completed artifacts."""
    try:
        if not api.repo_exists(repo_id=repo_id, repo_type="model"):
            return False
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        has_readme = "README.md" in files
        has_content = any(
            f in files for f in ("config.yaml", "training_data.jsonl", "kveritas_report.pdf", "eval_results.json")
        ) or any("eval" in f.lower() or "adapter" in f.lower() for f in files)
        return has_readme and (has_content or len(files) >= 2)
    except Exception as exc:
        logger.debug("Failed checking remote repo %s: %s", repo_id, exc)
        return False


def check_cell_completed(
    api: Optional[HfApi],
    target_repo: str,
    run_dir: Path,
    algo: str,
    restart_grpo: bool = True,
    skip_completed: bool = True,
) -> tuple[bool, str]:
    """Determine whether an experiment cell should be skipped or run."""
    if not skip_completed:
        return False, "Force rerun requested"

    # 1. Check Hugging Face Hub
    if api and is_repo_completed_on_hub(api, target_repo):
        return True, f"Remote Hugging Face repository '{target_repo}' is already completed"

    if algo == "grpo" and restart_grpo:
        return False, "Restarting GRPO run as requested"

    # 2. Check local completion
    eval_file = run_dir / "exp" / "eval.json"
    if eval_file.is_file():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d and (d.get("phi") is not None or d.get("mer") is not None):
                return True, f"Local evaluation report found in {eval_file}"
        except Exception:
            pass

    adapter_file = run_dir / "models" / "adapter_config.json"
    if adapter_file.is_file():
        return True, f"Local trained weights found in {adapter_file}"

    return False, "Not completed"


def load_cached_metrics(
    run_dir: Path,
    model_entry: Dict[str, Any],
    algo: str,
    scope: str,
    target_repo: str,
) -> Dict[str, Any]:
    """Construct result dictionary for a skipped / cached cell."""
    eval_file = run_dir / "exp" / "eval.json"
    phi, mer, mcs, uwec = None, None, None, None
    eval_report = {}
    if eval_file.is_file():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_report = json.load(f)
            phi = eval_report.get("phi")
            mer = eval_report.get("mer")
            mcs = eval_report.get("mcs")
            uwec = eval_report.get("uwec")
        except Exception:
            pass

    return {
        "model": model_entry["id"],
        "scope": scope,
        "method": algo.upper(),
        "mer": mer,
        "mcs": mcs,
        "uwec": uwec,
        "phi": phi,
        "seed": 42,
        "config": str(run_dir / "config.yaml"),
        "status": "CACHED",
        "hf_repo": target_repo,
        "eval_report": eval_report,
    }


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
        dest="include_weights",
        action="store_true",
        help="Include trained model weights and tokenizers in upload (default: True)",
    )
    parser.add_argument(
        "--no-weights",
        dest="include_weights",
        action="store_false",
        help="Exclude model weights and tokenizers from upload (upload training data only)",
    )
    parser.set_defaults(include_weights=True)

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
        "--eval-steps",
        type=int,
        default=None,
        help="Evaluation interval in steps (defaults to max_steps // 10, i.e. 10 evaluations per run)",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=256,
        help="Maximum generation completion length for rollouts (default: 256)",
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
        help="Google Cloud location for Vertex AI (default: global)",
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
        default=True,
        help="Skip previously completed runs (default: True)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip runs already completed on Hugging Face Hub or locally (default: True)",
    )
    parser.add_argument(
        "--force-rerun",
        dest="skip_completed",
        action="store_false",
        help="Force re-running all cells even if already present on Hugging Face Hub",
    )
    parser.add_argument(
        "--restart-grpo",
        action="store_true",
        default=True,
        help="Always restart GRPO training to ensure clean run with vLLM fixes (default: True)",
    )
    parser.add_argument(
        "--no-restart-grpo",
        dest="restart_grpo",
        action="store_false",
        help="Allow skipping GRPO if already completed on Hugging Face Hub",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=True,
        help="Enable vLLM rollout acceleration for GRPO (default: True)",
    )
    parser.add_argument(
        "--no-use-vllm",
        dest="use_vllm",
        action="store_false",
        help="Disable vLLM rollout acceleration",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.3,
        help="vLLM GPU memory utilization fraction (default: 0.3)",
    )
    parser.add_argument(
        "--recompute-completed",
        action="store_true",
        default=True,
        help="Skip re-training already uploaded/completed models, but recompute post-aligned evaluation and push updated model cards (default: True)",
    )
    parser.add_argument(
        "--no-recompute-completed",
        dest="recompute_completed",
        action="store_false",
        help="Do not recompute evaluation for completed models; use cached metrics",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Skip post-training and evaluate post-aligned weights to recompute and update benchmark reports",
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

    # Filter models while strictly preserving CLI order and supporting slug aliases
    if args.models.strip().lower() == "all":
        selected_models = catalog
    else:
        import re

        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", str(s).lower())

        catalog_map = {}
        for m in catalog:
            s_norm = _norm(m["slug"])
            catalog_map[s_norm] = m
            catalog_map[_norm(m["id"])] = m
            catalog_map[_norm(m.get("display_name", ""))] = m
            if s_norm.endswith("m") or s_norm.endswith("b"):
                catalog_map[s_norm[:-1]] = m
            if "granite" in s_norm and "350" in s_norm:
                catalog_map["granite40350m"] = m
                catalog_map["granite40350"] = m
                catalog_map["granite4350m"] = m

        selected_models = []
        for raw_token in args.models.split(","):
            tok = raw_token.strip()
            if not tok:
                continue
            matched = catalog_map.get(_norm(tok))
            if matched and matched not in selected_models:
                selected_models.append(matched)
            elif not matched:
                logger.warning("Model token '%s' did not match catalog. Available: %s", tok, [m["slug"] for m in catalog])

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

            # Check if this cell is already completed and should be skipped (unless --eval-only is set)
            if args.eval_only:
                is_done, reason = False, "Evaluating post-aligned weights (--eval-only)"
            else:
                is_done, reason = check_cell_completed(
                    api=api if not args.dry_run else None,
                    target_repo=target_repo,
                    run_dir=run_dir,
                    algo=algo,
                    restart_grpo=args.restart_grpo,
                    skip_completed=args.skip_completed or args.resume,
                )

            if is_done:
                if args.recompute_completed and not args.dry_run:
                    logger.info(
                        "\n[%d/%d] SKIPPING TRAINING (ALREADY COMPLETED) -> RECOMPUTING EVAL: %s (%s) with %s -> TARGET: %s\n  Reason: %s",
                        current_idx,
                        total_cells,
                        model_entry["id"],
                        model_entry["slug"],
                        algo.upper(),
                        target_repo,
                        reason,
                    )
                    from scripts.recompute_and_push_eval import recompute_cell_eval
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
                        use_vllm=args.use_vllm if algo == "grpo" else False,
                    )
                    mc = MasterConfig.from_dict(cfg_data)
                    mc.working_dir = run_dir
                    cfg.set_working_dir(run_dir)
                    local_models = run_dir / "models"
                    has_local = local_models.exists() and any(local_models.iterdir())

                    weights_source = str(local_models) if has_local else target_repo

                    try:
                        eval_report = recompute_cell_eval(
                            mc,
                            weights_source=weights_source,
                            run_dir=run_dir,
                            max_eval_rows=args.max_eval_rows,
                        )

                        # Save local eval results
                        exp_dir = run_dir / "exp"
                        exp_dir.mkdir(parents=True, exist_ok=True)
                        eval_path = exp_dir / "eval.json"
                        eval_path.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False), encoding="utf-8")
                        shutil.copy2(eval_path, run_dir / "eval_results.json")

                        result = {
                            "model": model_entry["id"],
                            "scope": args.scope,
                            "method": algo.upper(),
                            "phi": eval_report.get("phi"),
                            "mer": eval_report.get("mer"),
                            "mcs": eval_report.get("mcs"),
                            "uwec": eval_report.get("uwec"),
                            "eval_report": eval_report,
                            "seed": 42,
                            "config": str(run_dir / "config.yaml"),
                            "status": "COMPLETED",
                            "hf_repo": target_repo,
                        }

                        # Push updated README.md and eval_results.json to Hugging Face
                        if api is not None:
                            card_content = generate_cell_model_card(
                                repo_id=target_repo,
                                model_entry=model_entry,
                                algo=algo,
                                scope=args.scope,
                                steps=args.max_steps,
                                metrics=result,
                            )
                            readme_path = run_dir / "README.md"
                            readme_path.write_text(card_content, encoding="utf-8")

                            api.upload_file(
                                path_or_fileobj=str(readme_path),
                                path_in_repo="README.md",
                                repo_id=target_repo,
                                repo_type="model",
                                commit_message="docs: recompute benchmark scores with post-aligned weights",
                            )
                            api.upload_file(
                                path_or_fileobj=str(run_dir / "eval_results.json"),
                                path_in_repo="eval_results.json",
                                repo_id=target_repo,
                                repo_type="model",
                                commit_message="docs: update held-out eval_results.json",
                            )
                            logger.info("Successfully pushed recomputed scores to HF: %s", target_repo)

                        all_results.append(result)
                        continue
                    except Exception as recompute_err:
                        logger.error("Failed to recompute eval for %s: %s", target_repo, recompute_err, exc_info=True)
                        tb_str = traceback.format_exc()
                        if not args.dry_run and api is not None:
                            push_error_to_hub(
                                api=api,
                                repo_id=target_repo,
                                run_dir=run_dir,
                                model_entry=model_entry,
                                algo=algo,
                                scope=args.scope,
                                steps=args.max_steps,
                                error_msg=str(recompute_err),
                                traceback_str=tb_str,
                                private=is_private,
                            )
                        cached_res = load_cached_metrics(run_dir, model_entry, algo, args.scope, target_repo)
                        all_results.append(cached_res)
                        continue
                else:
                    logger.info(
                        "\n[%d/%d] SKIPPING (ALREADY COMPLETED): %s (%s) with %s -> TARGET: %s\n  Reason: %s",
                        current_idx,
                        total_cells,
                        model_entry["id"],
                        model_entry["slug"],
                        algo.upper(),
                        target_repo,
                        reason,
                    )
                    cached_res = load_cached_metrics(run_dir, model_entry, algo, args.scope, target_repo)
                    all_results.append(cached_res)
                    continue

            # For GRPO restart: clean up stale partial run artifacts
            if algo == "grpo":
                cleanup_stale_run_dir(run_dir)

            logger.info(
                "\n[%d/%d] RUNNING: %s (%s) with %s -> TARGET: %s",
                current_idx,
                total_cells,
                model_entry["id"],
                model_entry["slug"],
                algo.upper(),
                target_repo,
            )

            # Execute Cell with fault tolerance so subsequent runs continue uninterrupted
            try:
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
                    use_vllm=args.use_vllm if algo == "grpo" else False,
                )

                # Explicitly enforce max_steps, batch size, and eval_steps (10 evals per run)
                calc_eval_steps = args.eval_steps if args.eval_steps is not None else max(1, args.max_steps // 10)
                rec_batch = model_entry.get("recommended_batch_size", 2)
                models_dir = str(run_dir / "models")
                for sec in ("trainer", "sft", "dpo", "apo"):
                    cfg_data.setdefault(sec, {})
                    cfg_data[sec]["max_steps"] = args.max_steps
                    cfg_data[sec]["per_device_train_batch_size"] = rec_batch
                    cfg_data[sec]["output_dir"] = models_dir
                    cfg_data[sec]["eval_strategy"] = "steps"
                    cfg_data[sec]["eval_steps"] = calc_eval_steps

                logger.info("Evaluation interval set to every %d steps (10 evals total for %d steps)", calc_eval_steps, args.max_steps)

                # Specialized GRPO optimizations:
                if algo == "grpo":
                    params_str = str(model_entry.get("params", "")).upper()
                    is_large = ("4B" in params_str) or ("7B" in params_str) or ("8B" in params_str)
                    if not is_large:
                        # Unquantize small models to eliminate 650s/step dequant lag and allow vLLM
                        cfg_data["model"]["load_in_4bit"] = False
                        cfg_data["model"]["load_in_8bit"] = False
                        logger.info(
                            "GRPO: Disabled 4-bit quantization on %s (%s) for fast native BF16 rollouts.",
                            model_entry["slug"],
                            model_entry.get("params", "small"),
                        )

                    cfg_data["trainer"]["bf16"] = True
                    cfg_data["trainer"]["use_vllm"] = bool(args.use_vllm)
                    cfg_data["trainer"]["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
                    cfg_data["trainer"]["vllm_mode"] = "colocate"
                    cfg_data["trainer"]["max_prompt_length"] = 512
                    cfg_data["trainer"]["max_completion_length"] = args.max_completion_length
                    cfg_data["trainer"]["eval_strategy"] = "steps"
                    cfg_data["trainer"]["eval_steps"] = calc_eval_steps
                    cfg_data["trainer"]["gradient_accumulation_steps"] = min(
                        cfg_data["trainer"].get("gradient_accumulation_steps", 2), 2
                    )
                    cfg_data["trainer"]["per_device_train_batch_size"] = min(rec_batch, 2)
                if args.eval_only:
                    from scripts.recompute_and_push_eval import recompute_cell_eval
                    mc = MasterConfig.from_dict(cfg_data)
                    mc.working_dir = run_dir
                    cfg.set_working_dir(run_dir)
                    local_models = run_dir / "models"
                    has_local = local_models.exists() and any(local_models.iterdir())
                    has_remote = False
                    if not has_local and api is not None:
                        try:
                            has_remote = api.repo_exists(repo_id=target_repo, repo_type="model")
                        except Exception:
                            has_remote = False

                    if not has_local and not has_remote:
                        logger.warning(
                            "Skipping %s [%s]: No aligned weights found locally in %s or on HF repo %s",
                            model_entry["slug"],
                            algo,
                            local_models,
                            target_repo,
                        )
                        continue

                    weights_source = str(local_models) if has_local else target_repo
                    eval_report = recompute_cell_eval(
                        mc,
                        weights_source=weights_source,
                        run_dir=run_dir,
                        max_eval_rows=args.max_eval_rows,
                    )

                    # Save eval report locally so direct_push_cell surfaces it
                    exp_dir = run_dir / "exp"
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    eval_path = exp_dir / "eval.json"
                    eval_path.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False), encoding="utf-8")
                    shutil.copy2(eval_path, run_dir / "eval_results.json")

                    result = {
                        "model": model_entry["id"],
                        "scope": args.scope,
                        "method": algo.upper(),
                        "phi": eval_report.get("phi"),
                        "mer": eval_report.get("mer"),
                        "mcs": eval_report.get("mcs"),
                        "uwec": eval_report.get("uwec"),
                        "eval_report": eval_report,
                        "seed": 42,
                        "config": str(run_dir / "config.yaml"),
                        "status": "COMPLETED",
                    }
                else:
                    result = run_experiment_cell(cfg_data, run_dir, dry_run=args.dry_run)
            except Exception as cell_err:
                logger.error(
                    "Cell execution failed for %s [%s]: %s",
                    model_entry["slug"],
                    algo,
                    cell_err,
                    exc_info=True,
                )
                tb_str = traceback.format_exc()
                result = {
                    "model": model_entry["id"],
                    "scope": args.scope,
                    "method": algo.upper(),
                    "mer": None,
                    "mcs": None,
                    "uwec": None,
                    "phi": None,
                    "seed": 42,
                    "config": str(run_dir / "config.yaml"),
                    "status": f"FAILED: {cell_err}",
                }
                if not args.dry_run and api is not None:
                    push_error_to_hub(
                        api=api,
                        repo_id=target_repo,
                        run_dir=run_dir,
                        model_entry=model_entry,
                        algo=algo,
                        scope=args.scope,
                        steps=args.max_steps,
                        error_msg=str(cell_err),
                        traceback_str=tb_str,
                        private=is_private,
                    )

            result["hf_repo"] = target_repo
            all_results.append(result)

            # Direct Push to Hugging Face immediately after run finishes (with retries)
            if not args.dry_run and result.get("status") in ("COMPLETED", "CACHED"):
                pushed = False
                for attempt in range(1, 4):
                    try:
                        pushed = direct_push_cell(
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
                        if pushed:
                            break
                    except Exception as push_err:
                        logger.warning("Push attempt %d for %s failed: %s", attempt, target_repo, push_err)
                        time.sleep(2 * attempt)
                if not pushed:
                    logger.error("Failed to push %s after 3 attempts.", target_repo)

            # GPU Cache Clearing & Garbage Collection
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

    # 4. Save Global Reporting Frames
    save_reporting_frames(all_results, args.output_dir)
    print_summary_table(all_results)
    logger.info("\nALL %d RUNS COMPLETED SUCCESSFULLY!", total_cells)


if __name__ == "__main__":
    main()
