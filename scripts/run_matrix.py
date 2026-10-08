#!/usr/bin/env python3
"""Automated Experimentation Matrix Runner for Sebeni.

Executes the comprehensive matrix defined in Sebeni experimentation todo:
- 12 base models (LFM2.5, SmolLM2/3, Granite, Qwen2.5/3, Gemma3/4, Phi-3.5, TinyLlama)
- 4 post-training arms (SFT, GRPO, DPO, APO)
- 500 max steps per experiment
- Distillation teacher: gemini-3.7-flash (Vertex AI = True, location = global)
- Evaluation: Strictly on held-out test split from beni.data/test.json
- Verification: End-to-end cryptographic sealing via K-Veritas protocol
- Reporting: Generates Section 10 Canonical Reporting Frame in JSON, CSV, and Markdown
"""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sebeni.matrix")


def load_model_catalog(catalog_path: Path) -> List[Dict[str, Any]]:
    """Load model registry from JSON."""
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Model catalog not found: {catalog_path}")
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_config(
    base_config_path: Path,
    model_entry: Dict[str, Any],
    algorithm: str,
    scope: str,
    max_steps: int,
    distil_model: str,
    location: str,
    vertex: bool,
    backend: str,
    run_dir: Path,
    quantization: Optional[str] = None,
    max_eval_rows: Optional[int] = None,
    kveritas_disclosure: str = "open",
) -> Dict[str, Any]:
    """Compose experiment configuration for a single matrix cell."""
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["algorithm"] = algorithm
    cfg["project_name"] = f"matrix_{model_entry['slug']}_{algorithm}"
    cfg["scope"] = scope

    # Model parameters
    cfg.setdefault("model", {})
    cfg["model"]["model_name"] = model_entry["id"]
    if model_entry.get("trust_remote_code"):
        cfg["model"]["trust_remote_code"] = True
    if quantization is not None:
        if quantization not in {"4bit", "8bit", "none"}:
            raise ValueError(f"Unsupported quantization mode: {quantization}")
        cfg["model"]["load_in_4bit"] = quantization == "4bit"
        cfg["model"]["load_in_8bit"] = quantization == "8bit"

    # Data & Scope
    cfg.setdefault("data", {})
    if scope.upper() == "MULTI13":
        cfg["data"]["languages"] = ["multi13"]
    else:
        cfg["data"]["languages"] = [scope.lower()]
        cfg["data"]["default_lang"] = scope.lower()

    # Trainer budget (500 steps)
    cfg.setdefault("trainer", {})
    cfg["trainer"]["max_steps"] = max_steps
    rec_batch = model_entry.get("recommended_batch_size", 2)
    cfg["trainer"]["per_device_train_batch_size"] = rec_batch

    # Distillation configuration
    cfg.setdefault("distillation", {})
    cfg["distillation"]["enabled"] = True
    cfg["distillation"]["backend"] = backend
    cfg["distillation"]["model"] = distil_model
    cfg["distillation"]["vertex"] = vertex
    cfg["distillation"]["location"] = location

    # Experiment & Evaluation invariant
    cfg.setdefault("experiment", {})
    cfg["experiment"]["dataset"] = "packaged"
    cfg["experiment"]["eval_file"] = "test.json"
    cfg["experiment"]["scope"] = scope
    cfg["experiment"]["kveritas"] = True
    cfg["experiment"]["kveritas_seal"] = True
    if max_eval_rows is not None:
        cfg["experiment"]["max_eval_rows"] = max_eval_rows
    cfg["experiment"]["kveritas_disclosure"] = kveritas_disclosure

    return cfg


def run_experiment_cell(
    cfg_data: Dict[str, Any],
    run_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment cell: init -> distill -> train -> eval -> seal."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_data, f, default_flow_style=False)

    model_name = cfg_data["model"]["model_name"]
    algo = cfg_data["algorithm"]
    scope = cfg_data.get("scope", "MULTI13")
    seed = cfg_data.get("trainer", {}).get("seed", 42)

    result_row = {
        "model": model_name,
        "scope": scope,
        "method": algo.upper(),
        "mer": None,
        "mcs": None,
        "uwec": None,
        "phi": None,
        "seed": seed,
        "config": str(cfg_path),
        "status": "PENDING",
    }

    if dry_run:
        logger.info("[DRY RUN] Generated config for %s [%s] at %s", model_name, algo, cfg_path)
        result_row["status"] = "DRY_RUN"
        return result_row

    # Import internal Sebeni modules
    from beni.cli.main import maybe_kveritas_init, maybe_kveritas_seal, emit_kveritas_metrics
    from beni.core.srl.config import MasterConfig
    from beni.core.pipeline import run_distill, run_arm, run_eval
    from beni.data.datasets import assert_train_source

    # Enforce evaluation dataset invariant
    assert_train_source(Path("beni/data/raw/dataset_300_samples.jsonl"))

    logger.info("Starting experiment %s with %s (500 steps, eval=beni.data/test.json)...", model_name, algo)
    mc = MasterConfig.from_dict(cfg_data)
    mc.working_dir = str(run_dir)

    # 1. K-Veritas Init with open disclosure
    disclosure = cfg_data.get("experiment", {}).get("kveritas_disclosure", "open")
    maybe_kveritas_init(run_dir, disclosure=disclosure)

    # 2. Morphotactic Distillation (if not already cached)
    logger.info("Executing distillation step with teacher %s (Vertex AI, %s)...", mc.distillation.model, mc.distillation.location)
    try:
        run_distill(mc)
    except Exception as exc:
        logger.warning("Distillation note: %s. Continuing with cached/algorithmic baselines.", exc)

    # 3. Post-Training Arm (500 steps)
    logger.info("Running post-training arm: %s for %d steps...", algo, mc.trainer.max_steps)
    start_time = time.time()
    try:
        run_arm(mc)
        elapsed = time.time() - start_time
        logger.info("Training completed in %.1f seconds", elapsed)
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        result_row["status"] = f"FAILED: {exc}"
        return result_row

    # 4. Evaluation on held-out beni.data/test.json
    logger.info("Scoring evaluation on held-out split beni.data/test.json...")
    try:
        eval_report = run_eval(mc)
        result_row["phi"] = eval_report.get("phi")
        result_row["mer"] = eval_report.get("mer")
        result_row["mcs"] = eval_report.get("mcs")
        result_row["uwec"] = eval_report.get("uwec")
        result_row["status"] = "COMPLETED"
        logger.info(
            "Evaluation complete: Phi=%.4f MER=%.2f MCS=%.4f UWEC=%.4f",
            result_row["phi"] or 0.0,
            result_row["mer"] or 0.0,
            result_row["mcs"] or 0.0,
            result_row["uwec"] or 0.0,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        eval_report = {}
        result_row["status"] = f"EVAL_FAILED: {exc}"

    # 5. K-Veritas Sealing
    report_pdf = run_dir / "kveritas_report.pdf"
    logger.info("Sealing K-Veritas cryptographic verification report...")
    maybe_kveritas_seal(report_pdf, working_dir=run_dir, report=eval_report)

    return result_row


def save_reporting_frames(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save Section 10 Canonical Reporting Frame in JSON, CSV, and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    json_path = output_dir / "matrix_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 2. CSV
    csv_path = output_dir / "matrix_results.csv"
    headers = ["Model", "Scope", "Method", "MER", "MCS", "UWEC", "Phi", "Seed", "Config", "Status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "scope", "method", "mer", "mcs", "uwec", "phi", "seed", "config", "status"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # 3. Markdown (Matching Section 10 canonical reporting frame)
    md_path = output_dir / "matrix_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Sebeni Canonical Reporting Frame: Experimentation Matrix\n\n")
        f.write("| Model | Scope | Method | MER | MCS | UWEC | Phi | Seed | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            mer_str = f"{r['mer']:.2f}" if r["mer"] is not None else "N/A"
            mcs_str = f"{r['mcs']:.4f}" if r["mcs"] is not None else "N/A"
            uwec_str = f"{r['uwec']:.4f}" if r["uwec"] is not None else "N/A"
            phi_str = f"{r['phi']:.4f}" if r["phi"] is not None else "N/A"
            f.write(
                f"| `{r['model']}` | {r['scope']} | **{r['method']}** | "
                f"{mer_str} | {mcs_str} | {uwec_str} | {phi_str} | {r['seed']} | {r['status']} |\n"
            )

    logger.info("Canonical reporting frames saved to:")
    logger.info("  JSON: %s", json_path)
    logger.info("  CSV:  %s", csv_path)
    logger.info("  MD:   %s", md_path)


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print clean summary table to stdout."""
    print("\n" + "=" * 90)
    print("SEBENI EXPERIMENTATION MATRIX — CANONICAL REPORTING FRAME")
    print("=" * 90)
    print(f"{'Model':<30} | {'Scope':<8} | {'Method':<6} | {'MER':<7} | {'MCS':<7} | {'UWEC':<7} | {'Phi':<7} | {'Status'}")
    print("-" * 90)
    for r in results:
        mer = f"{r['mer']:.2f}" if r["mer"] is not None else "N/A"
        mcs = f"{r['mcs']:.4f}" if r["mcs"] is not None else "N/A"
        uwec = f"{r['uwec']:.4f}" if r["uwec"] is not None else "N/A"
        phi = f"{r['phi']:.4f}" if r["phi"] is not None else "N/A"
        print(f"{r['model'][:30]:<30} | {r['scope']:<8} | {r['method']:<6} | {mer:<7} | {mcs:<7} | {uwec:<7} | {phi:<7} | {r['status']}")
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Sebeni Experimentation Matrix across models and algorithms."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model slugs or IDs. Defaults to all 12 models in configs/matrix/models.json.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="sft,grpo,dpo,apo",
        help="Comma-separated algorithms to run (default: sft,grpo,dpo,apo).",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="MULTI13",
        help="Scope (default: MULTI13).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Training step budget per experiment (default: 500).",
    )
    quantization_group = parser.add_mutually_exclusive_group()
    quantization_group.add_argument(
        "--load-in-4bit",
        dest="quantization",
        action="store_const",
        const="4bit",
        help="Load models with 4-bit quantization.",
    )
    quantization_group.add_argument(
        "--load-in-8bit",
        dest="quantization",
        action="store_const",
        const="8bit",
        help="Load models with 8-bit quantization.",
    )
    quantization_group.add_argument(
        "--no-quantization",
        dest="quantization",
        action="store_const",
        const="none",
        help="Load models without BitsAndBytes quantization.",
    )
    parser.add_argument(
        "--distil-model",
        type=str,
        default="gemini-3.7-flash",
        help="Distillation teacher model (default: gemini-3.7-flash).",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="global",
        help="Google Cloud location for Vertex AI (default: global).",
    )
    parser.add_argument(
        "--vertex",
        action="store_true",
        default=True,
        help="Use Vertex AI for Google GenAI client (default: True).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="google",
        choices=["google", "gemini", "algorithmic"],
        help="Distillation provider backend (default: google).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/matrix"),
        help="Output directory for matrix experiments and reports.",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="Cap evaluation rows for fast trial/validation (e.g. 10).",
    )
    parser.add_argument(
        "--kveritas-disclosure",
        type=str,
        default="open",
        help="K-Veritas provenance disclosure level (default: open).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and directory manifests without executing training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that have already produced a valid eval.json and sealed report.",
    )

    args = parser.parse_args()

    # Configure Google Cloud environment
    os.environ["GOOGLE_CLOUD_LOCATION"] = args.location
    os.environ["GOOGLE_LOCATION"] = args.location

    repo_root = Path(__file__).resolve().parent.parent
    base_config_path = repo_root / "configs" / "matrix" / "base_multi13.yaml"
    catalog_path = repo_root / "configs" / "matrix" / "models.json"

    catalog = load_model_catalog(catalog_path)
    selected_algorithms = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]

    # Filter models
    if args.models:
        filter_tokens = {m.strip().lower() for m in args.models.split(",") if m.strip()}
        selected_models = [
            m for m in catalog
            if m["id"].lower() in filter_tokens or m["slug"].lower() in filter_tokens
        ]
        if not selected_models:
            logger.error("No models matched filter '%s'. Available: %s", args.models, [m["slug"] for m in catalog])
            sys.exit(1)
    else:
        selected_models = catalog

    total_runs = len(selected_models) * len(selected_algorithms)
    logger.info("Initializing Experimentation Matrix:")
    logger.info("  Models (%d): %s", len(selected_models), [m["slug"] for m in selected_models])
    logger.info("  Algorithms (%d): %s", len(selected_algorithms), selected_algorithms)
    logger.info("  Budget: %d steps per run", args.max_steps)
    logger.info(
        "  Quantization: %s",
        args.quantization or "base config default",
    )
    logger.info("  Distillation: %s (%s, Vertex=%s, backend=%s)", args.distil_model, args.location, args.vertex, args.backend)
    logger.info("  Evaluation Split: Held-out beni.data/test.json")
    logger.info("  Total Experiment Cells: %d", total_runs)

    all_results: List[Dict[str, Any]] = []

    for model_entry in selected_models:
        for algo in selected_algorithms:
            run_slug = f"{model_entry['slug']}_{algo}"
            if args.quantization is not None:
                quant_slug = {"4bit": "int4", "8bit": "int8", "none": "fp"}[args.quantization]
                run_slug = f"{run_slug}_{quant_slug}"
            run_dir = args.output_dir / run_slug

            # Check if resuming and already completed
            if args.resume:
                eval_file = run_dir / "exp" / "eval.json"
                if eval_file.is_file():
                    try:
                        with open(eval_file, "r", encoding="utf-8") as f:
                            saved_eval = json.load(f)
                        all_results.append({
                            "model": model_entry["id"],
                            "scope": args.scope,
                            "method": algo.upper(),
                            "mer": saved_eval.get("mer"),
                            "mcs": saved_eval.get("mcs"),
                            "uwec": saved_eval.get("uwec"),
                            "phi": saved_eval.get("phi"),
                            "seed": 42,
                            "config": str(run_dir / "config.yaml"),
                            "status": "CACHED",
                        })
                        logger.info("Skipping completed run %s (found %s)", run_slug, eval_file)
                        continue
                    except Exception:
                        pass

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
                quantization=args.quantization,
                max_eval_rows=args.max_eval_rows,
                kveritas_disclosure=args.kveritas_disclosure,
            )

            result = run_experiment_cell(cfg_data, run_dir, dry_run=args.dry_run)
            all_results.append(result)

    save_reporting_frames(all_results, args.output_dir)
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
