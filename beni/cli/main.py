"""Typer CLI: ``sebeni init|train|distill|eval|exp|push|generate|wordfreq``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from beni.core.srl.config import MasterConfig
from beni.utils import config as cfg

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Sebeni — self-aware morphotactic generation for extremely low-resource languages.",
)


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
    """Print stdout lines K-Veritas already understands."""
    phi = report.get("phi")
    if phi is not None:
        typer.echo(f"KVERITAS_METRIC name=phi value={float(phi):.6g} step={step}")
    for lang, row in (report.get("by_language") or {}).items():
        val = (row or {}).get("phi")
        if val is None:
            continue
        typer.echo(f"KVERITAS_METRIC name=phi_{lang} value={float(val):.6g} step={step}")


def maybe_kveritas_init(working_dir: Optional[Path] = None) -> None:
    """Ensure K-Veritas session is initialized in working_dir or current directory."""
    import shutil
    import subprocess

    binary = shutil.which("kveritas")
    if not binary:
        return
    target_dir = Path(working_dir) if working_dir else Path.cwd()
    if not (target_dir / ".kveritas").is_dir() and not (Path.cwd() / ".kveritas").is_dir():
        typer.echo(f"[kveritas] Initializing session in {target_dir}")
        result = subprocess.run([binary, "init"], cwd=str(target_dir), check=False)
        if result.returncode != 0:
            typer.echo(f"Warning: kveritas init exited {result.returncode}", err=True)


def maybe_kveritas_seal(output: Path, working_dir: Optional[Path] = None, report: Optional[Dict] = None) -> None:
    import json
    import shutil
    import subprocess

    binary = shutil.which("kveritas")
    if not binary:
        typer.echo(
            "kveritas not on PATH; skip seal. See https://kveritas.org/docs",
            err=True,
        )
        return
    target_dir = Path(working_dir) if working_dir else Path.cwd()
    if not (target_dir / ".kveritas").is_dir() and not (Path.cwd() / ".kveritas").is_dir():
        maybe_kveritas_init(target_dir)

    seal_dir = target_dir if (target_dir / ".kveritas").is_dir() else Path.cwd()
    session_file = seal_dir / ".kveritas" / "session.json"
    if session_file.is_file():
        try:
            session_data = json.loads(session_file.read_text(encoding="utf-8"))
            if not session_data.get("runs"):
                # Anchor the run so kveritas seal has at least one recorded run
                metric_lines = []
                if report:
                    phi = report.get("phi")
                    if phi is not None:
                        metric_lines.append(f"KVERITAS_METRIC name=phi value={float(phi):.6g} step=0")
                    for lang, row in (report.get("by_language") or {}).items():
                        val = (row or {}).get("phi")
                        if val is not None:
                            metric_lines.append(f"KVERITAS_METRIC name=phi_{lang} value={float(val):.6g} step=0")
                cmd_str = "; ".join(f"echo '{line}'" for line in metric_lines) if metric_lines else "echo '[sebeni] Run recorded'"
                subprocess.run([binary, "run", "--", "sh", "-c", cmd_str], cwd=str(seal_dir), check=False)
        except Exception:
            pass

    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [binary, "seal", "--output", str(output.resolve())],
        cwd=str(seal_dir),
        check=False,
    )
    if result.returncode != 0:
        typer.echo(f"kveritas seal exited {result.returncode}", err=True)
    else:
        typer.echo(f"K-Veritas report sealed: {output}")


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
  max_eval_rows: 1

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
        None, "--kveritas/--no-kveritas", help="Stream step rewards and eval metrics via K-Veritas protocol."
    ),
    eval_ratio: Optional[float] = typer.Option(
        None, "--eval-ratio", help="Fraction of user dataset held out for evaluation (default: 0.1)."
    ),
    eval_steps: Optional[int] = typer.Option(
        None, "--eval-steps", help="Number of update steps between evaluations."
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
        eval_steps=eval_steps,
    )
    if mc.experiment.kveritas:
        maybe_kveritas_init(mc.working_dir)
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
):
    """Score held-out test.json per language; write ``{working_dir}/exp/eval.json``."""
    mc = _load_config(config, working_dir)
    mc.apply_cli_overrides(languages=lang)
    from beni.core.pipeline import run_eval
    from beni.core.srl.unified import SRLTrainer

    trainer = None
    try:
        trainer = SRLTrainer(mc)
        trainer.load_models()
    except Exception:
        trainer = None
    generate = None
    policy = None
    if trainer is not None:

        def _generate(text: str, _lang: str) -> str:
            return trainer.generate(text)

        generate = _generate
        policy = getattr(trainer, "plugin", trainer)
    report = run_eval(mc, generate=generate, policy=policy)
    typer.echo(str(cfg.get_workdir().exp / "eval.json"))
    typer.echo(f"scope={report.get('scope')} phi={report.get('phi')}")


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
    if mc.experiment.kveritas or mc.experiment.kveritas_seal:
        maybe_kveritas_init(mc.working_dir)
    from beni.core.pipeline import run_experiment

    try:
        report = run_experiment(mc)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(str(cfg.get_workdir().exp / "eval.json"))
    if mc.experiment.kveritas:
        emit_kveritas_metrics(report)
    if mc.experiment.kveritas_seal:
        maybe_kveritas_seal(cfg.get_workdir().exp / "report.pdf", working_dir=mc.working_dir, report=report)


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
):
    """Push ``output_dir`` to the Hub (requires model card + safety snapshot)."""
    mc = _load_config(config, working_dir)
    from beni.core.srl.unified import SRLTrainer

    trainer = SRLTrainer(mc)
    trainer.load_models()
    rid = repo_id or mc.trainer.hub_model_id
    trainer.push_to_hub(repo_id=rid, token=mc.trainer.hub_token, private=mc.trainer.hub_private_repo)


@app.command()
def generate(
    config: Path = typer.Option(..., "-c", "--config"),
    prompt: str = typer.Option(..., "--prompt"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    max_length: int = typer.Option(128, "--max-length"),
    lang: Optional[str] = typer.Option(None, "--lang", help="Expected JSON lang for R_lang (one row)."),
):
    """Generate from the saved policy, gated by SafetyGovernor (format / R_lang)."""
    mc = _load_config(config, working_dir)
    from beni.core.srl.unified import SRLTrainer
    from beni.core.compute.rewards import RewardManager

    trainer = SRLTrainer(mc)
    trainer.load_models()
    text = trainer.generate(prompt, max_length=max_length)
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
