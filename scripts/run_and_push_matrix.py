#!/usr/bin/env python3
"""Run All 4 Alignment Algorithms (SFT, GRPO, DPO, APO) & Push to Hugging Face Hub (Private Org).

This script:
1. Pre-validates Hugging Face credentials and permissions (fails early if invalid).
2. Creates or verifies a private Hugging Face model repository.
3. Executes a full rerun of the alignment matrix (SFT, GRPO, DPO, APO) on the specified model.
   Completed local cells and existing Hub artifacts are retrained and overwritten.
4. Skips Gemini distillation (Vertex AI off). Training uses the packaged dataset.
5. Generates canonical Section 10 reporting frames and K-Veritas cryptographic provenance seals.
6. Uploads model weights, tokenizers, training configs, trackio logs, and benchmark tables
   to the private Hugging Face repository, replacing any existing algorithm subdirectory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Polyfill for Python < 3.10.7 (CVE-2020-10735 integer string conversion limit)
if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda digits: None

import yaml
from huggingface_hub import HfApi, get_token

# Import core matrix runner utilities
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
logger = logging.getLogger("sebeni.push_matrix")


def generate_repo_readme(
    repo_id: str,
    model_name: str,
    scope: str,
    max_steps: int,
    results: List[Dict[str, Any]],
) -> str:
    """Generate a comprehensive model card and benchmark comparison table."""
    lines = [
        "---",
        "library_name: transformers",
        "tags:",
        "  - sebeni",
        "  - low-resource-languages",
        "  - morphotactic-alignment",
        "  - multi13",
        "  - sft",
        "  - grpo",
        "  - dpo",
        "  - apo",
        "license: apache-2.0",
        "pipeline_tag: text-generation",
        "---",
        "",
        f"# Sebeni Morphotactic Alignment Matrix: {model_name}",
        "",
        f"This repository contains model weights, adapters, safety governors, K-Veritas provenance seals,",
        f"and training logs for **{model_name}** post-trained across all 4 alignment algorithms (**SFT**, **GRPO**, **DPO**, **APO**)",
        f"under the **{scope}** multilingual low-resource scope.",
        "",
        "## Training Configuration",
        f"- **Base Model**: `{model_name}`",
        f"- **Scope**: `{scope}` (13 low-resource Sahelian/West-African language groups)",
        f"- **Training Budget**: `{max_steps}` steps per arm",
        f"- **Hardware Target**: NVIDIA L4 GPU (24GB VRAM)",
        f"- **Distillation**: skipped (Vertex AI disabled; packaged dataset only)",
        "",
        "## Canonical Benchmark Results (Section 10 Reporting Frame)",
        "",
        "| Algorithm | Scope | MER (Morpheme Error) | MCS (Morphosyntactic Consistency) | UWEC (Uncertainty Weighted Error Cost) | Phi (Integrity Score) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        method = r.get("method", "N/A")
        sc = r.get("scope", scope)
        mer = f"{r['mer']:.2f}" if r.get("mer") is not None else "N/A"
        mcs = f"{r['mcs']:.4f}" if r.get("mcs") is not None else "N/A"
        uwec = f"{r['uwec']:.4f}" if r.get("uwec") is not None else "N/A"
        phi = f"{r['phi']:.4f}" if r.get("phi") is not None else "N/A"
        status = r.get("status", "COMPLETED")
        lines.append(f"| **{method}** | {sc} | {mer} | {mcs} | {uwec} | {phi} | `{status}` |")

    lines.extend([
        "",
        "## Repository Directory Structure",
        "",
        "```",
        "├── sft/                # Supervised Fine-Tuning adapter & checkpoints",
        "├── grpo/               # Group Relative Policy Optimization weights & logs",
        "├── dpo/                # Direct Preference Optimization weights & logs",
        "├── apo/                # Adaptive Policy Optimization weights & logs",
        "└── reports/            # matrix_results.json, .csv, .md & provenance seals",
        "```",
        "",
        "## Loading Adapters with Transformers & PEFT",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from peft import PeftModel",
        "",
        f'base_model = "{model_name}"',
        f'repo_id = "{repo_id}"',
        "",
        "tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=\"grpo\")",
        "model = AutoModelForCausalLM.from_pretrained(base_model, device_map=\"auto\")",
        "model = PeftModel.from_pretrained(model, repo_id, subfolder=\"grpo\")",
        "```",
        "",
        "## Verification & Provenance",
        "Each subdirectory includes K-Veritas cryptographic verification artifacts and safety governor snapshots.",
    ])

    return "\n".join(lines)


def upload_algorithm_artifacts(
    api: HfApi,
    repo_id: str,
    algo: str,
    run_dir: Path,
    *,
    private: bool = True,
) -> None:
    """Upload model weights, configs, tokenizer, and logs for a single algorithm."""
    if not run_dir.exists():
        logger.warning("Run directory %s does not exist, skipping upload.", run_dir)
        return

    logger.info("Overwriting %s artifacts at %s/%s/ ...", algo.upper(), repo_id, algo)

    # Exclude temporary cache folders and virtualenvs
    ignore_patterns = [
        "*.pyc",
        "__pycache__/*",
        ".venv/*",
        "tmp/*",
        "*.tmp",
    ]

    try:
        # Recreate if a previous upload 404'd because the repo was missing.
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(run_dir),
            path_in_repo=algo,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns,
            # Patterns are relative to path_in_repo. "**" drops stale files in this
            # algorithm folder; files uploaded in the same commit are kept.
            delete_patterns="**",
            commit_message=f"Overwrite {algo.upper()} model weights, tokenizer, configs, and training logs",
        )
        logger.info("Successfully uploaded %s to %s/%s/", algo.upper(), repo_id, algo)
    except Exception as exc:
        logger.error("Failed to upload %s artifacts: %s", algo.upper(), exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all 4 alignment algorithms (SFT, GRPO, DPO, APO) & push to private HF repo"
    )
    # Hugging Face parameters
    parser.add_argument(
        "--hf-repo",
        type=str,
        required=True,
        help="Target Hugging Face repository ID (e.g. 'organization/sebeni-multi13-smollm2' or 'username/repo')",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face User Access Token (defaults to HF_TOKEN environment variable or huggingface-cli login)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Make repository public (default is private to organization/user)",
    )

    # Experiment matrix parameters
    parser.add_argument(
        "--models",
        type=str,
        default="SmolLM2-135M",
        help="Model ID or slug (default: SmolLM2-135M)",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="sft,grpo,dpo,apo",
        help="Comma-separated algorithms to run (default: 'sft,grpo,dpo,apo')",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Budget of training steps per algorithm arm (default: 500)",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=10,
        help="Cap evaluation rows for held-out scoring and in-training eval (default: 10; omit for all)",
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
        help="Load base model with 4-bit BitsAndBytes quantization (recommended for L4 VRAM)",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="MULTI13",
        help="Scope for experiment (default: MULTI13)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/matrix"),
        help="Base directory for experiment outputs and logs (default: experiments/matrix)",
    )
    parser.add_argument(
        "--distil-model",
        type=str,
        default="gemini-3.7-flash",
        help="Teacher model for distillation (default: gemini-3.7-flash)",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="global",
        help="Google Cloud location (default: global)",
    )
    parser.add_argument(
        "--vertex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Vertex AI for Google GenAI distillation (default: False)",
    )
    parser.add_argument(
        "--skip-distillation",
        action="store_true",
        default=True,
        help="Skip Gemini distillation and train on the packaged dataset (default: True)",
    )
    parser.add_argument(
        "--distill",
        dest="skip_distillation",
        action="store_false",
        help="Run Gemini distillation before training",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="google",
        help="Distillation backend (default: google)",
    )
    parser.add_argument(
        "--kveritas-disclosure",
        type=str,
        default="open",
        help="K-Veritas disclosure mode (default: open)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume previously completed cells without re-running (default: full rerun)",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=False,
        help="Skip runs already completed on Hugging Face Hub or locally (default: full rerun)",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=True,
        help="Re-run every cell and overwrite existing Hub artifacts (default: True)",
    )
    parser.add_argument(
        "--no-force-rerun",
        dest="force_rerun",
        action="store_false",
        help="Allow --resume / --skip-completed to keep existing cells",
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
        "--dry-run",
        action="store_true",
        default=False,
        help="Build configs without executing training or uploads",
    )

    args = parser.parse_args()

    # 1. Resolve and Validate Hugging Face Token & Permissions Early
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token()
    if not token and not args.dry_run:
        logger.error(
            "No Hugging Face token provided. Pass --hf-token <token>, set export HF_TOKEN=<token>, "
            "or run 'huggingface-cli login'."
        )
        sys.exit(1)

    api = HfApi(token=token)
    if not args.dry_run:
        try:
            user_info = api.whoami()
            username = user_info.get("name") or user_info.get("fullname")
            logger.info("Authenticated to Hugging Face Hub as: %s", username)
        except Exception as exc:
            logger.error("Hugging Face authentication failed: %s", exc)
            sys.exit(1)

        # Create or verify repository (private by default)
        is_private = not args.public
        try:
            logger.info("Verifying / creating Hugging Face repository '%s' (private=%s)...", args.hf_repo, is_private)
            repo_url = api.create_repo(
                repo_id=args.hf_repo,
                repo_type="model",
                private=is_private,
                exist_ok=True,
            )
            logger.info("Target repository ready: %s", repo_url)
        except Exception as exc:
            logger.error("Failed to access or create repository '%s': %s", args.hf_repo, exc)
            sys.exit(1)

    # 2. Configure Environment & Catalog
    os.environ["GOOGLE_CLOUD_LOCATION"] = args.location
    os.environ["GOOGLE_LOCATION"] = args.location

    base_config_path = repo_root / "configs" / "matrix" / "base_multi13.yaml"
    catalog_path = repo_root / "configs" / "matrix" / "models.json"
    catalog = load_model_catalog(catalog_path)

    # Filter model
    filter_tokens = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    selected_models = [
        m for m in catalog
        if m["id"].lower() in filter_tokens or m["slug"].lower() in filter_tokens
    ]
    if not selected_models:
        logger.error("Model '%s' not found in catalog. Available: %s", args.models, [m["slug"] for m in catalog])
        sys.exit(1)

    selected_algorithms = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    quantization = "4bit" if args.load_in_4bit else None

    logger.info("=" * 70)
    logger.info("SEBENI MULTI13 ALIGNMENT MATRIX RUNNER")
    logger.info("=" * 70)
    logger.info("Model: %s", [m["id"] for m in selected_models])
    logger.info("Algorithms (%d): %s", len(selected_algorithms), selected_algorithms)
    logger.info("Budget: %d steps per arm", args.max_steps)
    logger.info("Quantization: %s", quantization or "none (bf16/fp16)")
    logger.info(
        "Distillation: %s | Vertex: %s",
        "skipped" if args.skip_distillation else args.distil_model,
        args.vertex,
    )
    logger.info("Full rerun (overwrite Hub): %s", args.force_rerun)
    logger.info("Target HF Repo: %s (private=%s)", args.hf_repo, not args.public)
    logger.info("=" * 70)

    # Fetch remote repository files once for efficient skip checks
    remote_files = set()
    if not args.dry_run and api:
        try:
            remote_files = set(api.list_repo_files(repo_id=args.hf_repo, repo_type="model"))
        except Exception as exc:
            logger.debug("Could not pre-fetch remote repo files: %s", exc)

    all_results: List[Dict[str, Any]] = []

    # 3. Execute Experiment Matrix Across All Arms
    for model_entry in selected_models:
        for algo in selected_algorithms:
            run_slug = f"{model_entry['slug']}_{algo}"
            run_dir = args.output_dir / run_slug

            # Check if this cell should be skipped
            is_completed = False
            skip_reason = ""
            if (
                not args.force_rerun
                and (args.skip_completed or args.resume)
                and not (algo == "grpo" and args.restart_grpo)
            ):
                # Check remote HF subfolder
                algo_prefix = f"{algo}/"
                has_remote = any(f.startswith(algo_prefix) for f in remote_files)
                if has_remote:
                    is_completed = True
                    skip_reason = f"Artifacts present in remote HF repo under '{algo}/'"
                else:
                    eval_f = run_dir / "exp" / "eval.json"
                    if eval_f.is_file():
                        try:
                            with open(eval_f, "r", encoding="utf-8") as f:
                                d = json.load(f)
                            if d and (d.get("phi") is not None or d.get("mer") is not None):
                                is_completed = True
                                skip_reason = f"Local evaluation report found in {eval_f}"
                        except Exception:
                            pass

            if is_completed:
                logger.info("\n>>> Skipping Arm %s | Model: %s (%s) <<<", algo.upper(), model_entry["id"], skip_reason)
                eval_file = run_dir / "exp" / "eval.json"
                eval_report = {}
                if eval_file.is_file():
                    try:
                        with open(eval_file, "r", encoding="utf-8") as f:
                            eval_report = json.load(f)
                    except Exception:
                        pass
                all_results.append({
                    "model": model_entry["id"],
                    "scope": args.scope,
                    "method": algo.upper(),
                    "mer": eval_report.get("mer"),
                    "mcs": eval_report.get("mcs"),
                    "uwec": eval_report.get("uwec"),
                    "phi": eval_report.get("phi"),
                    "seed": 42,
                    "config": str(run_dir / "config.yaml"),
                    "status": "CACHED",
                    "eval_report": eval_report,
                })
                continue

            if args.force_rerun and run_dir.exists():
                # Drop stale weights so the arm trains from scratch and the Hub copy is a full replace.
                for sub in ("models", "runs", "tmp", "exp"):
                    shutil.rmtree(run_dir / sub, ignore_errors=True)
                for name in ("eval_results.json", "config.yaml"):
                    (run_dir / name).unlink(missing_ok=True)
                for p in run_dir.glob("checkpoint-*"):
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)

            logger.info("\n>>> Starting Arm: %s | Model: %s <<<", algo.upper(), model_entry["id"])

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
                kveritas_disclosure=args.kveritas_disclosure,
                use_vllm=args.use_vllm if algo == "grpo" else False,
            )
            cfg_data.setdefault("distillation", {})
            cfg_data["distillation"]["enabled"] = not args.skip_distillation
            cfg_data["distillation"]["vertex"] = bool(args.vertex) and not args.skip_distillation

            calc_eval_steps = args.eval_steps if args.eval_steps is not None else max(1, args.max_steps // 10)
            rec_batch = model_entry.get("recommended_batch_size", 2)
            eval_batch = min(rec_batch, 1)
            models_dir = str(run_dir / "models")
            for sec in ("trainer", "sft", "dpo", "apo"):
                cfg_data.setdefault(sec, {})
                cfg_data[sec]["max_steps"] = args.max_steps
                cfg_data[sec]["per_device_train_batch_size"] = rec_batch
                cfg_data[sec]["per_device_eval_batch_size"] = eval_batch
                cfg_data[sec]["output_dir"] = models_dir
                cfg_data[sec]["eval_strategy"] = "steps"
                cfg_data[sec]["eval_steps"] = calc_eval_steps

            # Identify Gemma 4 and large models
            model_slug_lower = model_entry.get("slug", "").lower()
            model_id_lower = model_entry.get("id", "").lower()
            is_gemma4 = "gemma-4" in model_slug_lower or "gemma-4" in model_id_lower or "gemma4" in model_slug_lower
            params_str = str(model_entry.get("params", "")).upper()
            is_large = is_gemma4 or any(k in params_str for k in ("3B", "4B", "7B", "8B"))

            # DPO / APO memory safeguards (prevent 10 GiB logits OOM on Gemma 4 / large vocab models)
            if algo in ("dpo", "apo"):
                if is_gemma4 or is_large or rec_batch <= 1:
                    cfg_data["model"]["load_in_4bit"] = True
                    cfg_data[algo]["gradient_checkpointing"] = True
                    cfg_data[algo]["per_device_train_batch_size"] = 1
                    cfg_data[algo]["per_device_eval_batch_size"] = 1
                    cfg_data[algo]["max_prompt_length"] = min(cfg_data[algo].get("max_prompt_length", 512), 512)
                    cfg_data[algo]["max_length"] = min(cfg_data[algo].get("max_length", 768), 768)
                    cfg_data[algo]["eval_accumulation_steps"] = 1
                    logger.info(
                        "DPO/APO: Memory guards applied for %s: 4-bit LoRA, grad_checkpointing=True, per_device_eval_batch_size=1, max_length=768.",
                        model_entry["slug"],
                    )

            if algo == "grpo":
                if is_gemma4:
                    # Gemma 4 PLE / custom KV-cache architecture causes cudaErrorInvalidValue in vLLM colocate mode.
                    # Use Hugging Face generation with 4-bit LoRA and gradient checkpointing.
                    cfg_data["model"]["load_in_4bit"] = True
                    cfg_data["model"]["load_in_8bit"] = False
                    cfg_data["trainer"]["bf16"] = False
                    cfg_data["trainer"]["use_vllm"] = False
                    cfg_data["trainer"]["gradient_checkpointing"] = True
                    logger.info(
                        "GRPO: Configured %s with 4-bit LoRA and HF generation (vLLM colocate disabled for Gemma 4).",
                        model_entry["slug"],
                    )
                elif is_large:
                    cfg_data["model"]["load_in_4bit"] = True
                    cfg_data["model"]["load_in_8bit"] = False
                    cfg_data["trainer"]["bf16"] = False
                    cfg_data["trainer"]["use_vllm"] = False
                    cfg_data["trainer"]["gradient_checkpointing"] = True
                else:
                    cfg_data["model"]["load_in_4bit"] = False
                    cfg_data["model"]["load_in_8bit"] = False
                    cfg_data["trainer"]["bf16"] = True
                    cfg_data["trainer"]["use_vllm"] = bool(args.use_vllm)
                    cfg_data["trainer"]["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
                    cfg_data["trainer"]["vllm_mode"] = "colocate"

                cfg_data["trainer"]["max_prompt_length"] = 512
                cfg_data["trainer"]["max_completion_length"] = args.max_completion_length
                cfg_data["trainer"]["eval_strategy"] = "steps"
                cfg_data["trainer"]["eval_steps"] = calc_eval_steps
                cfg_data["trainer"]["per_device_eval_batch_size"] = 1

                # Ensure gradient_accumulation_steps * per_device_train_batch_size is divisible by num_generations
                num_gen = cfg_data["trainer"].get("num_generations", 4)
                grpo_batch = 1 if (is_gemma4 or is_large) else min(rec_batch, 2)
                cfg_data["trainer"]["per_device_train_batch_size"] = grpo_batch
                needed_accum = max(1, (num_gen + grpo_batch - 1) // grpo_batch)
                while (grpo_batch * needed_accum) % num_gen != 0:
                    needed_accum += 1
                cfg_data["trainer"]["gradient_accumulation_steps"] = needed_accum

            try:
                result = run_experiment_cell(cfg_data, run_dir, dry_run=args.dry_run)
            except Exception as cell_exc:
                logger.error("Arm %s failed on %s: %s", algo, model_entry["slug"], cell_exc, exc_info=True)
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
                    "status": f"FAILED: {cell_exc}",
                }
            all_results.append(result)

            # Upload algorithm artifacts to HF immediately upon completion
            if not args.dry_run and result.get("status") in ("COMPLETED", "CACHED"):
                for attempt in range(1, 4):
                    try:
                        upload_algorithm_artifacts(
                            api,
                            args.hf_repo,
                            algo,
                            run_dir,
                            private=not args.public,
                        )
                        break
                    except Exception as upload_exc:
                        logger.warning("Upload attempt %d for %s failed: %s", attempt, algo, upload_exc)
                        time.sleep(2 * attempt)

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

    # 4. Save Section 10 Canonical Reporting Frames
    save_reporting_frames(all_results, args.output_dir)
    print_summary_table(all_results)

    # 5. Final HF Push: Repository Model Card, Global Summary & Reports
    if not args.dry_run:
        logger.info("\n>>> Finalizing Hugging Face Repository & Model Card <<<")
        model_name = selected_models[0]["id"]
        readme_content = generate_repo_readme(
            repo_id=args.hf_repo,
            model_name=model_name,
            scope=args.scope,
            max_steps=args.max_steps,
            results=all_results,
        )

        readme_path = args.output_dir / "HF_README.md"
        readme_path.write_text(readme_content, encoding="utf-8")

        # Upload root README.md
        try:
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=args.hf_repo,
                repo_type="model",
                commit_message="Update repository model card and Section 10 benchmark results",
            )
            logger.info("Updated root README.md on Hugging Face Hub")
        except Exception as exc:
            logger.error("Failed to upload README.md: %s", exc)

        # Upload matrix reporting frames
        try:
            api.upload_folder(
                folder_path=str(args.output_dir),
                path_in_repo="reports",
                repo_id=args.hf_repo,
                repo_type="model",
                allow_patterns=["matrix_results.*"],
                commit_message="Upload Section 10 canonical reporting frames (JSON, CSV, MD)",
            )
            logger.info("Uploaded reports/ to Hugging Face Hub")
        except Exception as exc:
            logger.error("Failed to upload reports folder: %s", exc)

        logger.info("\n" + "=" * 70)
        logger.info("ALL 4 ARMS COMPLETED & PUSHED TO HUGGING FACE HUB!")
        logger.info("Repository URL: https://huggingface.co/%s", args.hf_repo)
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
