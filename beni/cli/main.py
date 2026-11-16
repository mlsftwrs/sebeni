"""Typer CLI: ``sebeni init|train|distill|eval|exp|push|generate|wordfreq``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import typer

from beni.core.srl.config import MasterConfig
from beni.utils import config as cfg

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Sebeni — self-aware morphotactic generation for extremely low-resource languages.",
)


@app.callback()
def _main() -> None:
    """Sebeni — self-aware morphotactic generation for extremely low-resource languages."""
    from beni.core.kveritas import restore_caller_cwd

    restore_caller_cwd()


def _load_config(
    config: Optional[Path],
    working_dir: Optional[Path] = None,
) -> MasterConfig:
    if config is not None:
        return MasterConfig.from_yaml(config, working_dir=working_dir)
    mc = MasterConfig()
    mc.apply_workdir(cli_path=working_dir)
    return mc


def _packaged_exp_yaml() -> Path:
    pkg = Path(__file__).resolve().parents[1] / "data" / "exp.yaml"
    repo = Path(__file__).resolve().parents[2] / "configs" / "exp.yaml"
    if repo.is_file():
        return repo
    return pkg


def emit_kveritas_metrics(report: Dict, step: int = 0) -> None:
    """Print the report's Φ / MER / MCS / UWEC (scope and per language) as ``KV`` metric lines."""
    from beni.core import kveritas as kv

    for key in kv.CLAIM_METRICS:
        kv.emit_metric(key, report.get(key), step=step)
    for lang, row in (report.get("by_language") or {}).items():
        for key in kv.CLAIM_METRICS:
            kv.emit_metric(f"{key}_{lang}", (row or {}).get(key), step=step)


def maybe_kveritas_init(
    working_dir: Optional[Path] = None, disclosure: str = "open", local: bool = False
) -> bool:
    """Ensure a K-Veritas session exists in ``working_dir`` (or the cwd).

    Returns False when ``kveritas init`` fails (e.g. the attestation server is
    unreachable); True when a session exists or the binary is missing.
    """
    import shutil
    import subprocess

    binary = shutil.which("kveritas")
    if not binary:
        return True
    target_dir = Path(working_dir) if working_dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    if not (target_dir / ".kveritas").is_dir():
        mode = "local key" if local else "attestation server"
        typer.echo(f"[kveritas] Initializing session in {target_dir} with disclosure={disclosure} ({mode})")
        cmd = [binary, "init"]
        if disclosure:
            cmd.extend(["--disclosure", disclosure])
        if local:
            cmd.append("--local")
        result = subprocess.run(cmd, cwd=str(target_dir), check=False)
        if result.returncode != 0:
            typer.echo(f"Warning: kveritas init exited {result.returncode}", err=True)
            return False
    return True


def maybe_kveritas_seal(output: Path, working_dir: Optional[Path] = None, report: Optional[Dict] = None) -> None:
    """Seal an existing K-Veritas session into ``output``.

    Never creates a run: a session without recorded runs is refused, because a
    report must cover the experiment that actually ran (``sebeni exp`` / ``train``
    / ``eval`` with K-Veritas enabled run themselves under ``kveritas run``).
    """
    import shutil
    import subprocess

    del report  # kept for API compatibility; claims come from the monitored run
    binary = shutil.which("kveritas")
    if not binary:
        typer.echo("kveritas not on PATH; skip seal. See https://kveritas.org/docs", err=True)
        return
    target_dir = Path(working_dir) if working_dir else Path.cwd()
    session_file = target_dir / ".kveritas" / "session.json"
    runs = []
    if session_file.is_file():
        try:
            runs = json.loads(session_file.read_text(encoding="utf-8")).get("runs") or []
        except (OSError, ValueError):
            runs = []
    if not runs:
        typer.echo(
            f"Error: no recorded K-Veritas run in {target_dir}; nothing to seal. "
            "Run `sebeni exp` / `train` / `eval` with experiment.kveritas enabled first.",
            err=True,
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([binary, "seal", "--output", str(output.resolve())], cwd=str(target_dir), check=False)
    if result.returncode != 0:
        typer.echo(f"kveritas seal exited {result.returncode}", err=True)
    else:
        typer.echo(f"K-Veritas report sealed: {output}")


def _kveritas_files(mc: MasterConfig, config: Optional[Path]) -> List[str]:
    """Inputs hashed before and after the monitored run: config, train data, held-out data."""
    from beni.core import kveritas as kv

    source = mc.data.source
    eval_source = mc.experiment.eval_source
    return kv.evidence_files(
        [
            config,
            source if isinstance(source, (list, tuple)) else [source],
            eval_source if isinstance(eval_source, (list, tuple)) else [eval_source],
            cfg.DATA_DIR / "test.json",
            cfg.DATA_DIR / "raw" / "dataset_300_samples.jsonl",
        ]
    )


def _maybe_reexec_under_kveritas(mc: MasterConfig, config: Optional[Path], seal: bool) -> None:
    """Re-run this command once as the monitored subprocess of ``kveritas run``.

    No-op inside the monitored child, when K-Veritas is off, or when the
    ``kveritas`` binary is missing (a warning is printed). Otherwise exits with
    the monitored run's return code. ``seal`` writes ``{working_dir}/exp/report.pdf``
    after the whole command finishes.
    """
    import shutil
    import subprocess
    import sys

    from beni.core import kveritas as kv

    if kv.under_kveritas() or not (mc.experiment.kveritas or seal):
        return
    binary = shutil.which("kveritas")
    if not binary:
        typer.echo("Warning: kveritas not on PATH; running without K-Veritas.", err=True)
        return
    workdir = Path(mc.working_dir or cfg.get_workdir().root)
    if not maybe_kveritas_init(
        workdir, disclosure=mc.experiment.kveritas_disclosure, local=mc.experiment.kveritas_local
    ):
        typer.echo(
            "Error: could not start a K-Veritas session, so the run would not be recorded. "
            "Set experiment.kveritas_local: true to sign with a local key, fix the attestation "
            "server, or rerun with --no-kveritas.",
            err=True,
        )
        raise typer.Exit(code=1)
    report = workdir / "exp" / "report.pdf" if seal else None
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
    cmd = kv.kveritas_command(binary, sys.argv[1:], seal=report, files=_kveritas_files(mc, config))
    typer.echo(f"[kveritas] Monitored run: {' '.join(cmd)}")
    code = subprocess.call(cmd, cwd=str(workdir), env=kv.child_env())
    if code == 0 and report is not None:
        typer.echo(f"K-Veritas report sealed: {report}")
    raise typer.Exit(code=code)


def default_config_yaml(langs: List[str], working_dir: str) -> str:
    """Scaffold YAML for ``sebeni init``."""
    from beni.core.language import resolve_scope

    resolved = resolve_scope(langs)
    codes = resolved.languages
    default = codes[0]
    langs_yaml = "[" + ", ".join(codes) + "]"
    slug = "multi13" if resolved.label == "MULTI13" else "-".join(codes)
    return f"""# Sebeni MasterConfig — https://seben.robotsmali.org/docs
# working_dir is relocatable: CLI -w and SEBENI_HOME / SEBENI_WORKING_DIR also apply.
# Distillation runs once, then one arm (sft | grpo | dpo | apo) reads frozen G/D.
# Tune trainer/model hyperparams below or via `sebeni train --lr ...`.
project_name: sebeni-{slug}
algorithm: grpo          # sft | grpo | dpo | apo
working_dir: {working_dir}

model:
  model_name: HuggingFaceTB/SmolLM2-135M
  # ref_model_name: null
  load_in_4bit: true
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  # lora_target_modules: [q_proj, v_proj, k_proj, o_proj]

data:
  default_lang: {default}
  languages: {langs_yaml}
  source: null             # path, glob, list of paths, or Hugging Face dataset id
  scheme: completion       # completion | preference | online_group
  eval_ratio: 0.1          # in-training hold-out fraction (data.eval_rows sets a row count instead)

trainer:
  framework: torch         # torch | jax
  learning_rate: 5.0e-6
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_steps: 10
  num_train_epochs: 1.0
  logging_steps: 1
  save_steps: 50
  max_grad_norm: 0.1
  beta: 0.1
  num_generations: 4
  num_iterations: 1
  temperature: 0.9
  top_p: 1.0
  top_k: 50
  warmup_ratio: 0.0
  warmup_steps: 0
  weight_decay: 0.0
  lr_scheduler_type: cosine
  seed: 42
  optim: adamw_torch
  bf16: false
  fp16: false
  gradient_checkpointing: false
  dataloader_num_workers: 0
  use_cpu: false
  # output_dir defaults to {{working_dir}}/models

distillation:
  enabled: true
  backend: algorithmic     # algorithmic | gguf | google | openai | groq | together
  model: gemini-2.5-flash
  # vertex: true           # ADC / Vertex AI; omit to auto-select when ADC is present
  tau: 0.5
  hitl: false

wordfreq:
  raw_inputs: null         # defaults to packaged beni/data/raw

experiment:
  dataset: packaged        # dataset_300_samples.jsonl; test.json stays eval-only
  freeze_resources: true
  scope: {resolved.label}
  max_eval_rows: null      # held-out test.json rows per language; null scores all of them

reward:
  format_weight: 0.1
  morph_weight: 0.4
  rule_weight: 0.4
  lang_weight: 0.1

safety:
  enabled: true
  require_model_card: true
  require_safety_snapshot: true
"""


@app.command()
def init(
    lang: List[str] = typer.Option(
        ["multi13"],
        "--lang",
        help="multi13, all, one group code, or a comma-separated list (bam,mku). bbo is the outlier scope.",
    ),
    working_dir: Path = typer.Option(
        Path("./runs/sebeni-001"),
        "-w",
        "--working-dir",
        help="Run directory (created if missing).",
    ),
):
    """Write config.yaml and the workdir layout (data/, models/, runs/, exp/, runtime/)."""
    from beni.core.language import resolve_scope

    resolved = resolve_scope(lang)
    wd = cfg.set_working_dir(working_dir, ensure=True)
    config_path = Path(wd.root) / "config.yaml"
    rel = str(wd.root)
    config_path.write_text(default_config_yaml(resolved.languages, rel), encoding="utf-8")
    typer.echo(f"Wrote {config_path}")
    typer.echo(f"Working dir {wd.root}")
    typer.echo(f"Scope: {resolved.label}")
    typer.echo("Languages: " + ", ".join(resolved.languages))
    if resolved.dropped:
        typer.echo("Dropped from this scope: " + ", ".join(resolved.dropped))
    typer.echo("Next: sebeni distill -c config.yaml")


@app.command()
def train(
    config: Path = typer.Option(..., "-c", "--config", help="YAML or JSON MasterConfig."),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    algorithm: Optional[str] = typer.Option(
        None, "--algorithm", help="sft | grpo | dpo | apo. Overrides the config."
    ),
    lang: Optional[List[str]] = typer.Option(
        None,
        "--lang",
        help="Scope override: multi13, all, one code, or a comma-separated list.",
    ),
    hitl: Optional[bool] = typer.Option(None, "--hitl/--no-hitl", help="HITL on initial distill."),
    lr: Optional[float] = typer.Option(None, "--lr", help="Optimizer learning rate."),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "--per-device-train-batch-size", help="Per-device train batch size."
    ),
    grad_accum: Optional[int] = typer.Option(
        None, "--grad-accum", "--gradient-accumulation-steps"
    ),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
    epochs: Optional[float] = typer.Option(None, "--epochs", "--num-train-epochs"),
    beta: Optional[float] = typer.Option(None, "--beta", help="KL / DPO β."),
    num_generations: Optional[int] = typer.Option(None, "--num-generations", help="GRPO group size G."),
    max_prompt_length: Optional[int] = typer.Option(None, "--max-prompt-length"),
    max_completion_length: Optional[int] = typer.Option(None, "--max-completion-length"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    warmup_ratio: Optional[float] = typer.Option(None, "--warmup-ratio"),
    warmup_steps: Optional[int] = typer.Option(None, "--warmup-steps"),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    save_steps: Optional[int] = typer.Option(None, "--save-steps"),
    optim: Optional[str] = typer.Option(None, "--optim"),
    lr_scheduler: Optional[str] = typer.Option(None, "--lr-scheduler", help="e.g. cosine, linear."),
    lora_r: Optional[int] = typer.Option(None, "--lora-r"),
    lora_alpha: Optional[int] = typer.Option(None, "--lora-alpha"),
    lora_dropout: Optional[float] = typer.Option(None, "--lora-dropout"),
    load_in_4bit: Optional[bool] = typer.Option(None, "--load-in-4bit/--no-load-in-4bit"),
    use_peft: Optional[bool] = typer.Option(None, "--peft/--no-peft"),
    bf16: Optional[bool] = typer.Option(None, "--bf16/--no-bf16"),
    fp16: Optional[bool] = typer.Option(None, "--fp16/--no-fp16"),
    gradient_checkpointing: Optional[bool] = typer.Option(
        None, "--grad-checkpoint/--no-grad-checkpoint"
    ),
    use_cpu: Optional[bool] = typer.Option(None, "--use-cpu/--no-use-cpu"),
    kveritas: Optional[bool] = typer.Option(
        None,
        "--kveritas/--no-kveritas",
        help="Run the whole command under `kveritas run` and stream KV metrics. Default: experiment.kveritas.",
    ),
    kveritas_seal: Optional[bool] = typer.Option(
        None,
        "--kveritas-seal/--no-kveritas-seal",
        help="Run under `kveritas run --seal` and write {working_dir}/exp/report.pdf. Default: experiment.kveritas_seal.",
    ),
    eval_ratio: Optional[float] = typer.Option(
        None, "--eval-ratio", help="Fraction of user dataset held out for evaluation (default: 0.1)."
    ),
    eval_rows: Optional[int] = typer.Option(
        None, "--eval-rows", help="Rows of the train data held out for in-training eval. Overrides --eval-ratio."
    ),
    eval_steps: Optional[int] = typer.Option(
        None, "--eval-steps", help="Number of update steps between evaluations."
    ),
    use_vllm: Optional[bool] = typer.Option(
        None, "--use-vllm/--no-use-vllm", help="Enable vLLM rollout acceleration for GRPO."
    ),
    vllm_gpu_memory_utilization: Optional[float] = typer.Option(
        None, "--vllm-gpu-memory-utilization", help="GPU memory fraction reserved for vLLM rollouts (e.g. 0.3)."
    ),
):
    """Train one arm. Distills and freezes G/D first when that has not been done."""
    mc = _load_config(config, working_dir)
    if algorithm:
        mc.algorithm = algorithm.strip().lower()
    mc.apply_cli_overrides(
        languages=lang,
        learning_rate=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
        epochs=epochs,
        beta=beta,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=temperature,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        seed=seed,
        save_steps=save_steps,
        optim=optim,
        use_cpu=use_cpu,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        load_in_4bit=load_in_4bit,
        use_peft=use_peft,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        lr_scheduler_type=lr_scheduler,
        hitl=hitl,
        kveritas=kveritas,
        eval_ratio=eval_ratio,
        eval_rows=eval_rows,
        eval_steps=eval_steps,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    seal = mc.experiment.kveritas_seal if kveritas_seal is None else kveritas_seal
    _maybe_reexec_under_kveritas(mc, config, seal=seal)
    from beni.core.pipeline import run_arm

    run_arm(mc)


@app.command()
def distill(
    config: Path = typer.Option(..., "-c", "--config"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    lang: Optional[List[str]] = typer.Option(None, "--lang"),
    hitl: Optional[bool] = typer.Option(None, "--hitl/--no-hitl"),
):
    """Upstream SAMPG distillation. Writes frozen G/D and does not train a policy."""
    mc = _load_config(config, working_dir)
    mc.apply_cli_overrides(languages=lang, hitl=hitl)
    from beni.core.pipeline import run_distill

    run_distill(mc)


@app.command()
def eval(
    config: Path = typer.Option(..., "-c", "--config"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    lang: Optional[List[str]] = typer.Option(None, "--lang"),
    weights: Optional[str] = typer.Option(
        None,
        "--weights",
        help="Trained adapter/model directory or Hub repo id. Default: the arm's output dir ({working_dir}/models).",
    ),
    eval_file: Optional[str] = typer.Option(
        None,
        "--eval-file",
        help="Held-out rows to score (path, glob, or HF dataset id). Overrides experiment.eval_source; "
        "default is the packaged test.json.",
    ),
    kveritas: Optional[bool] = typer.Option(
        None,
        "--kveritas/--no-kveritas",
        help="Run the eval under `kveritas run` and claim its metrics. Default: experiment.kveritas.",
    ),
    kveritas_seal: Optional[bool] = typer.Option(
        None,
        "--kveritas-seal/--no-kveritas-seal",
        help="Run under `kveritas run --seal` and write {working_dir}/exp/report.pdf. Default: experiment.kveritas_seal.",
    ),
):
    """Score the trained policy on held-out rows per language; write ``{working_dir}/exp/eval.json``.

    Loads the trained weights (``--weights``, default the arm's output dir,
    ``{working_dir}/models``) and decodes greedily with the training prompt.
    Without trained weights it scores the base model and says so.
    """
    mc = _load_config(config, working_dir)
    mc.apply_cli_overrides(languages=lang)
    if eval_file:
        mc.experiment.eval_source = eval_file
    if kveritas is not None:
        mc.experiment.kveritas = kveritas
    seal = mc.experiment.kveritas_seal if kveritas_seal is None else kveritas_seal
    _maybe_reexec_under_kveritas(mc, config, seal=seal)
    from beni.core.pipeline import run_eval

    trainer = _load_policy(mc, weights, require=bool(weights))

    def _generate(text: str, _lang: str) -> str:
        return trainer.analyze(text)

    report = run_eval(
        mc,
        generate=_generate,
        policy=getattr(trainer, "plugin", trainer),
        weights=trainer.loaded_weights,
    )
    typer.echo(str(cfg.get_workdir().exp / "eval.json"))
    typer.echo(f"scope={report.get('scope')} phi={report.get('phi')}")


def _load_policy(mc: MasterConfig, weights: Optional[str], require: bool):
    """Load the trained policy for eval / push / generate, or exit with a clear error."""
    from beni.core.pipeline import default_weights_dir, load_policy

    try:
        trainer = load_policy(mc, weights=weights, require=require)
    except (OSError, RuntimeError) as exc:
        typer.echo(f"Error: could not load the policy: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if trainer.loaded_weights:
        typer.echo(f"Loaded trained weights: {trainer.loaded_weights}")
    else:
        typer.echo(
            f"Warning: no trained weights in {default_weights_dir(mc)}; using the base model "
            f"{mc.model.model_name}. Pass --weights to choose a checkpoint.",
            err=True,
        )
    return trainer


def _preset_yaml(name: str) -> Path:
    token = str(name or "multi13").strip().lower()
    if token in {"all", "multi13"}:
        token = "multi13"
    root = Path(__file__).resolve().parents[2] / "configs" / "presets" / f"{token}.yaml"
    if root.is_file():
        return root
    packaged = Path(__file__).resolve().parents[1] / "data" / "presets" / f"{token}.yaml"
    return packaged if packaged.is_file() else root


def _run_exp(
    config: Optional[Path],
    working_dir: Optional[Path],
    preset: Optional[str] = None,
    algorithm: Optional[str] = None,
    lang: Optional[List[str]] = None,
) -> None:
    cfg_path = _preset_yaml(preset) if preset else (config or _preset_yaml("multi13"))
    if not Path(cfg_path).is_file():
        cfg_path = config or _packaged_exp_yaml()
    if not Path(cfg_path).is_file():
        typer.echo(f"experiment config not found: {cfg_path}", err=True)
        raise typer.Exit(code=2)
    mc = _load_config(Path(cfg_path), working_dir)
    if algorithm:
        mc.algorithm = algorithm.strip().lower()
    if lang:
        mc.apply_cli_overrides(languages=lang)
    mc.experiment.dataset = "packaged"
    mc.data.source = None
    _maybe_reexec_under_kveritas(mc, Path(cfg_path), seal=bool(mc.experiment.kveritas_seal))
    from beni.core.pipeline import run_experiment

    try:
        report = run_experiment(mc)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    del report  # Φ / MER / MCS / UWEC are claimed by run_eval (KV claim lines)
    typer.echo(str(cfg.get_workdir().exp / "eval.json"))


@app.command("exp")
def exp(
    config: Optional[Path] = typer.Option(
        None,
        "-c",
        "--config",
        help="YAML MasterConfig. Defaults to the MULTI13 preset. data.source is ignored.",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="multi13 or single. Overrides -c when set.",
    ),
    algorithm: Optional[str] = typer.Option(
        None, "--algorithm", help="sft | grpo | dpo | apo."
    ),
    lang: Optional[List[str]] = typer.Option(
        None,
        "--lang",
        help="Scope override. Required in spirit for --preset single (default bam in the preset).",
    ),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
):
    """Distill, train one arm on the packaged experiment jsonl, then evaluate test.json."""
    _run_exp(config, working_dir, preset=preset, algorithm=algorithm, lang=lang)


@app.command("experiment")
def experiment(
    config: Optional[Path] = typer.Option(
        None,
        "-c",
        "--config",
        help="YAML MasterConfig. Defaults to the MULTI13 preset. data.source is ignored.",
    ),
    preset: Optional[str] = typer.Option(None, "--preset", help="multi13 or single."),
    algorithm: Optional[str] = typer.Option(None, "--algorithm", help="sft | grpo | dpo | apo."),
    lang: Optional[List[str]] = typer.Option(None, "--lang"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
):
    """Alias of ``sebeni exp``."""
    _run_exp(config, working_dir, preset=preset, algorithm=algorithm, lang=lang)


@app.command()
def push(
    config: Path = typer.Option(..., "-c", "--config"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    repo_id: Optional[str] = typer.Option(None, "--repo-id"),
    write_card: bool = typer.Option(
        False, "--write-card", help="Re-render README.md from the eval safety snapshot before pushing."
    ),
    weights: Optional[str] = typer.Option(
        None,
        "--weights",
        help="Trained adapter/model directory or Hub repo id. Default: the arm's output dir ({working_dir}/models).",
    ),
):
    """Push the trained policy to the Hub.

    Requires trained weights, plus README.md and an eval safety_snapshot.json
    next to them: run ``sebeni eval`` first. ``--write-card`` re-renders the
    card from that snapshot. A LoRA run pushes the adapter.
    """
    mc = _load_config(config, working_dir)
    trainer = _load_policy(mc, weights, require=True)
    rid = repo_id or mc.trainer.hub_model_id
    try:
        trainer.push_to_hub(
            repo_id=rid,
            token=mc.trainer.hub_token,
            private=mc.trainer.hub_private_repo,
            output_dir=trainer.weights_dir,
            write_card=write_card,
        )
    except PermissionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Pushed {trainer.weights_dir} to {rid}")


@app.command()
def generate(
    config: Path = typer.Option(..., "-c", "--config"),
    prompt: str = typer.Option(..., "--prompt"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    max_new_tokens: Optional[int] = typer.Option(
        None,
        "--max-new-tokens",
        "--max-length",
        help="Decoding budget. Default: experiment.eval_max_new_tokens.",
    ),
    lang: Optional[str] = typer.Option(None, "--lang", help="Expected JSON lang for R_lang (one row)."),
    weights: Optional[str] = typer.Option(
        None,
        "--weights",
        help="Trained adapter/model directory or Hub repo id. Default: the arm's output dir ({working_dir}/models).",
    ),
):
    """Analyze one sentence with the trained policy, gated by SafetyGovernor (format / R_lang).

    ``--prompt`` is the sentence; it is wrapped in the training prompt and
    decoded greedily.
    """
    mc = _load_config(config, working_dir)
    from beni.core.compute.rewards import RewardManager

    trainer = _load_policy(mc, weights, require=bool(weights))
    text = trainer.analyze(prompt, max_new_tokens=max_new_tokens)
    rm = RewardManager(reward_config=mc.reward)
    fmt = rm.reward_format([text])
    expected = lang or (mc.languages()[0] if mc.languages() else "bam")
    lang_scores = rm.reward_lang([text], language=[expected])
    if fmt[0] == 0.0 and mc.safety.format_invalid_blocks_update:
        typer.echo("Warning: completion failed R_format (JSON + tokens).", err=True)
    if lang_scores[0] == 0.0 and mc.safety.require_language:
        typer.echo(f"Warning: completion failed R_lang (expected {expected}).", err=True)
    typer.echo(text)


@app.command()
def wordfreq(
    config: Path = typer.Option(..., "-c", "--config"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    lang: Optional[List[str]] = typer.Option(None, "--lang"),
):
    """Build DabaX frequency maps from configured raw text inputs."""
    mc = _load_config(config, working_dir)
    mc.apply_cli_overrides(languages=lang)
    from beni.core.wordfreq import count_raw_inputs

    raw_inputs = mc.wordfreq.raw_inputs or mc.data.source or (cfg.DATA_DIR / "raw")
    base = Path(getattr(mc, "_config_file_dir", Path.cwd()))
    values = raw_inputs if isinstance(raw_inputs, list) else [raw_inputs]
    raw_inputs = [
        str(Path(value) if Path(value).is_absolute() else base / str(value))
        for value in values
    ]
    reports = count_raw_inputs(
        raw_inputs,
        languages=mc.data.languages,
        default_lang=mc.data.default_lang or "bam",
        encoding=mc.data.encoding,
    )
    root = cfg.get_workdir().exp / "wordfreq"
    index = {"languages": list(reports.keys()), "by_language": {}}
    last_path = root
    for group, report in reports.items():
        path = report.write(root / group)
        last_path = path
        index["by_language"][group] = {
            "path": str(path),
            "n_tokens": report.n_tokens,
            "n_sentences": report.n_sentences,
            "checkpoint_id": report.checkpoint_id,
            "n_misses": sum(report.misses.values()),
        }
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "wordfreq.json"
    summary.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(str(summary if reports else last_path))


if __name__ == "__main__":
    app()
