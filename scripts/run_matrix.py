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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Polyfill for Python < 3.10.7 (CVE-2020-10735 integer string conversion limit)
if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda digits: None

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
    use_vllm: Optional[bool] = None,
    kveritas_local_key: Optional[str] = None,
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

    # Trainer parameters
    if use_vllm is not None:
        cfg.setdefault("trainer", {})
        cfg["trainer"]["use_vllm"] = use_vllm

    # Data & Scope
    cfg.setdefault("data", {})
    if scope.upper() == "MULTI13":
        cfg["data"]["languages"] = ["multi13"]
    else:
        cfg["data"]["languages"] = [scope.lower()]
        cfg["data"]["default_lang"] = scope.lower()

    # Trainer budget across all algorithms (GRPO uses trainer, SFT uses sft, DPO uses dpo, APO uses apo)
    rec_batch = model_entry.get("recommended_batch_size", 2)
    eval_batch = min(rec_batch, 1)
    models_dir = str(run_dir / "models")
    eval_interval = max(1, max_steps // 10)
    for section in ("trainer", "sft", "dpo", "apo"):
        cfg.setdefault(section, {})
        cfg[section]["max_steps"] = max_steps
        cfg[section]["per_device_train_batch_size"] = rec_batch
        cfg[section]["per_device_eval_batch_size"] = eval_batch
        cfg[section]["output_dir"] = models_dir
        cfg[section]["eval_strategy"] = "steps"
        cfg[section]["eval_steps"] = eval_interval

    model_slug_lower = model_entry.get("slug", "").lower()
    model_id_lower = model_entry.get("id", "").lower()
    is_gemma4 = "gemma-4" in model_slug_lower or "gemma-4" in model_id_lower or "gemma4" in model_slug_lower
    params_str = str(model_entry.get("params", "")).upper()
    is_large = is_gemma4 or any(k in params_str for k in ("3B", "4B", "7B", "8B"))

    if is_gemma4 or is_large or rec_batch <= 1:
        cfg.setdefault("model", {})
        cfg["model"]["load_in_4bit"] = True
        for section in ("trainer", "sft", "dpo", "apo"):
            cfg[section]["gradient_checkpointing"] = True
            cfg[section]["per_device_eval_batch_size"] = 1
        if "dpo" in cfg:
            cfg["dpo"]["max_length"] = min(cfg["dpo"].get("max_length", 768), 768)
            cfg["dpo"]["max_prompt_length"] = min(cfg["dpo"].get("max_prompt_length", 512), 512)
            cfg["dpo"]["eval_accumulation_steps"] = 1
        if "apo" in cfg:
            cfg["apo"]["max_length"] = min(cfg["apo"].get("max_length", 768), 768)
            cfg["apo"]["max_prompt_length"] = min(cfg["apo"].get("max_prompt_length", 512), 512)
            cfg["apo"]["eval_accumulation_steps"] = 1

    # Ensure GRPO generation_batch_size (per_device_train_batch_size * gradient_accumulation_steps) is divisible by num_generations
    if algorithm.lower() == "grpo":
        if is_gemma4:
            cfg["trainer"]["use_vllm"] = False
            cfg["trainer"]["bf16"] = False
        num_gen = cfg.get("trainer", {}).get("num_generations", 4)
        grpo_batch = 1 if (is_gemma4 or is_large) else min(rec_batch, 2)
        cfg["trainer"]["per_device_train_batch_size"] = grpo_batch
        needed_accum = max(1, (num_gen + grpo_batch - 1) // grpo_batch)
        while (grpo_batch * needed_accum) % num_gen != 0:
            needed_accum += 1
        cfg["trainer"]["gradient_accumulation_steps"] = needed_accum

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
    cfg["experiment"]["scope"] = scope
    cfg["experiment"]["kveritas"] = True
    cfg["experiment"]["kveritas_seal"] = True
    if max_eval_rows is not None:
        cfg["experiment"]["max_eval_rows"] = max_eval_rows
    cfg["experiment"]["kveritas_disclosure"] = kveritas_disclosure
    if kveritas_local_key:
        # Local signing (no attestation server): `kveritas init --local` + `seal --local-key`.
        cfg["experiment"]["kveritas_local"] = True
        cfg["experiment"]["kveritas_local_key"] = str(Path(kveritas_local_key).expanduser().resolve())

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

    # One `sebeni exp` subprocess per cell. With experiment.kveritas / kveritas_seal on
    # (set by build_config), that command runs itself under `kveritas run --seal`,
    # so each cell gets one monitored run covering distill, train and eval, and a
    # sealed {run_dir}/exp/report.pdf.
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    cmd = [
        sys.executable, "-m", "beni.cli.main", "exp",
        "-c", str(cfg_path.resolve()),
        "-w", str(run_dir.resolve()),
        "--algorithm", algo,
    ]
    logger.info("Starting cell %s [%s]: %s", model_name, algo, " ".join(cmd))
    start_time = time.time()
    # Never read a previous attempt's results or report for this cell.
    for stale in ("exp/eval.json", "exp/report.pdf", "exp/report.pdf.kvbundle.zip", "kveritas_report.pdf",
                  "kveritas_report.pdf.kvbundle.zip"):
        (run_dir / stale).unlink(missing_ok=True)
    proc = subprocess.run(cmd, env=env, check=False)
    logger.info("Cell finished in %.1f seconds (exit %d)", time.time() - start_time, proc.returncode)

    eval_path = run_dir / "exp" / "eval.json"
    eval_report: Dict[str, Any] = {}
    if eval_path.is_file():
        try:
            eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.error("Unreadable %s: %s", eval_path, exc)
    for key in ("phi", "mer", "mcs", "uwec"):
        result_row[key] = eval_report.get(key)
    result_row["eval_report"] = eval_report
    if proc.returncode != 0:
        result_row["status"] = f"FAILED: exit {proc.returncode}"
    elif not eval_report:
        result_row["status"] = "EVAL_FAILED: no exp/eval.json"
    else:
        result_row["status"] = "COMPLETED"

    # Keep the historical location used by the push scripts.
    sealed = run_dir / "exp" / "report.pdf"
    if sealed.is_file():
        shutil.copy2(sealed, run_dir / "kveritas_report.pdf")
        bundle = sealed.with_name(sealed.name + ".kvbundle.zip")
        if bundle.is_file():
            shutil.copy2(bundle, run_dir / "kveritas_report.pdf.kvbundle.zip")
        result_row["kveritas_report"] = str(run_dir / "kveritas_report.pdf")
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
            extrasaction="ignore",
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
        "--kveritas-local-key",
        type=str,
        default=None,
        help="Sign K-Veritas reports locally with this RSA private key (PEM) instead of the attestation server.",
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
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=None,
        help="Enable vLLM rollout acceleration for GRPO.",
    )
    parser.add_argument(
        "--no-use-vllm",
        dest="use_vllm",
        action="store_false",
        help="Disable vLLM rollout acceleration.",
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
                use_vllm=args.use_vllm,
                kveritas_local_key=args.kveritas_local_key,
            )

            result = run_experiment_cell(cfg_data, run_dir, dry_run=args.dry_run)
            all_results.append(result)

    save_reporting_frames(all_results, args.output_dir)
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
