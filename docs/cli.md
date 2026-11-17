# CLI and config

Public docs: [https://mlsftwrs.github.io/sebeni](https://mlsftwrs.github.io/sebeni).

Console script: `sebeni` (Typer). Autodoc of the Typer `app` object is
intentionally omitted — it has no stable inspect signature and used to break
the API pages.

```
sebeni init     --lang multi13 -w ./runs/multi13
sebeni init     --lang bam -w ./runs/bam
sebeni distill  -c ./runs/multi13/config.yaml
sebeni train    -c ./runs/multi13/config.yaml --algorithm grpo --lr 1e-5
sebeni eval     -c ./runs/multi13/config.yaml
sebeni exp      --preset multi13 --algorithm sft -w ./runs/multi13-sft
sebeni exp      --preset single --lang bbo --algorithm sft -w ./runs/bbo-sft
sebeni wordfreq -c config.yaml
sebeni push     -c config.yaml --repo-id mlsftwrs/sebeni-bam-grpo
sebeni generate -c config.yaml --prompt "Aw ka kɛnɛ wa?" --lang bam
```

YAML overlays `MasterConfig` dataclasses (no second config system). Presets
ship in the package: `beni/data/presets/multi13.yaml`,
`beni/data/presets/single.yaml`; an unknown `--preset` exits 2 and lists them.
Unknown config keys log a warning ("Ignoring unknown config key …").
`configs/exp.yaml` is a separate MULTI13 config, not an alias: its reward
weights are 0.1/0.4/0.4/0.1 (format/morph/rule/lang), the preset's are
0.2/0.3/0.3/0.2. Full trainer / LoRA tables: [Hyperparameters](hyperparams.md).

## Commands

| Command | Purpose | Important flags |
| --- | --- | --- |
| `sebeni init` | Write `config.yaml` + workdir (`data/`, `models/`, `runs/`, `exp/`, `runtime/`) | `--lang` (default `multi13`), `-w` (default `./runs/sebeni-001`) |
| `sebeni distill` | Upstream SAMPG once per language; freeze G and D | `-c`, `-w`, `--lang`, `--hitl/--no-hitl` |
| `sebeni train` | One arm against frozen G and D (distills first if `data/resources/frozen.json` is missing) | `-c`, `--algorithm`, `-w`, `--lang`, `--lr`, `--max-steps`, `--eval-ratio`, `--eval-rows`, `--eval-steps`, `--use-vllm`, `--kveritas/--no-kveritas`, `--kveritas-seal`; full list in [Hyperparameters](hyperparams.md#train-flags) |
| `sebeni exp` | Distill, one arm, held-out eval (`experiment` alias); trains on `data.source` from `-c`, else packaged `dataset_300_samples.jsonl` | `--preset multi13\|single`, `--algorithm`, `--lang`, `-c`, `-w`, `--kveritas/--no-kveritas`, `--kveritas-seal` |
| `sebeni eval` | Held-out Φ / MER / MCS / UWEC of the trained weights on packaged `test.json` (or `experiment.eval_source`) → `{working_dir}/exp/eval.json` | `-c`, `-w`, `--lang`, `--weights`, `--eval-file`, `--kveritas/--no-kveritas`, `--kveritas-seal` |
| `sebeni wordfreq` | Surfaces / lemmas / morphemes / stages per language | `-c`, `-w`, `--lang` |
| `sebeni generate` | Analyse one sentence with the trained weights; warn on `R_format` / `R_lang` | `-c`, `--prompt`, `--lang`, `--weights`, `--max-new-tokens` (alias `--max-length`; default `experiment.eval_max_new_tokens`, 512) |
| `sebeni push` | Hub upload of the trained weights (default the arm's output dir); needs an eval snapshot for those weights | `-c`, `-w`, `--repo-id` (else `trainer.hub_model_id`), `--weights`, `--write-card` |

`--lang multi13` or `--lang all` selects the 13 canonical group codes.
`--lang bam` is one language. `--lang bam,mku` is a comma-separated list.
`--lang bbo` is the outlier and is not mixed into MULTI13.

Setting `--epochs` without `--max-steps` sets `max_steps: -1` so Hugging Face
runs by epoch count.

`sebeni exp`, `train` and `eval` take `--kveritas/--no-kveritas` and
`--kveritas-seal/--no-kveritas-seal` (defaults: `experiment.kveritas` /
`experiment.kveritas_seal`; `--no-kveritas` turns sealing off too). With
K-Veritas on, the command re-launches itself once as the monitored process of
`kveritas run`; do not wrap it in `kveritas run` yourself. Seals:
`exp/report.pdf` (`exp`), `exp/report_train.pdf` (`train`),
`exp/report_eval.pdf` (`eval`).

`sebeni eval`, `generate` and `push` load the trained weights from `--weights`
(a directory or Hub repo id; default the arm's output dir,
`{working_dir}/models`). A LoRA adapter is attached to a fresh base model; a
full checkpoint is loaded directly. Without trained weights, `eval` warns and
scores the base model (`weights: null` in `eval.json`); `push` refuses.
Decoding is greedy, uses the same prompt as training and stops at end-of-turn
or after `experiment.eval_max_new_tokens` (512). `eval --lang` only filters rows.

`sebeni eval` scores packaged `beni/data/test.json` unless
`experiment.eval_source` or `--eval-file` names other held-out rows (path, glob,
list, or HF dataset id, like `data.source`). `sebeni push` needs `README.md` and
a `safety_snapshot.json` written by `sebeni eval` for exactly the pushed weight
files; nothing is auto-written, and `--write-card` re-renders the card from the
eval snapshot.

## `eval.json` shape

`mer` and `uwec` are costs (lower is better). `mcs` is the token-stage match
fraction (higher is better). `phi` is a macro mean over languages; `mer`,
`mcs`, `uwec` pool morphemes / tokens over the scope. A metric is `null` when
nothing was decoded and scored. `experiment.max_eval_rows` caps the rows scored
per language; the default `null` (or `0`) scores all of them.

```json
{
  "model": "Qwen/Qwen2.5-0.5B",
  "algorithm": "sft",
  "scope": "MULTI13",
  "languages": ["bam", "mku"],
  "phi": 0.63,
  "mer": 0.21,
  "mcs": 0.82,
  "uwec": 1.04,
  "n_morphemes": 1840,
  "n_tokens": 920,
  "beta": 0.1,
  "eps": 1e-8,
  "n_scored": 3,
  "n_skipped": 0,
  "reference_model": "Qwen/Qwen2.5-0.5B",
  "weights": "./runs/multi13-sft/models",
  "eval_source": "packaged:test.json",
  "max_eval_rows": null,
  "tau": 0.5,
  "by_language": {
    "bam": {
      "phi": 0.71,
      "mer": 0.19,
      "mcs": 0.84,
      "uwec": 1.02,
      "n_sentences": 3,
      "n_scored": 3,
      "n_skipped": 0,
      "n_morphemes": 1200,
      "n_tokens": 600,
      "checkpoint_id": "baseline_v3",
      "language": "bam",
      "scope": "MULTI13"
    }
  }
}
```

Per language, `n_sentences` is the number of reference sentences DabaX parsed
and `n_scored` the number of decoded rows that were scored; rows whose
reference has no morphemes are skipped and counted in `n_skipped`.
`weights` is the scored weights directory or repo (`null` for the base model).
`checkpoint_id`
names the G, D used (`baseline_vN`, `baseline`, `packaged`, or `none`), not a
policy checkpoint.

## MasterConfig fields

| Field | Meaning |
| --- | --- |
| `project_name` | Trackio / run name |
| `algorithm` | `sft` / `grpo` (default) / `dpo` / `apo` |
| `working_dir` | Relocatable root |
| `model` | `ModelConfig` (base, ref, LoRA rank/alpha/dropout, 4-bit / 8-bit, `flax_model_name`) |
| `processor` | `DabaXProcessorConfig` (runtime dir, baseline dir, tokenizer) |
| `data` | `DataConfig` (`source`, `scheme`, `languages`, `default_lang`, `eval_ratio`, `eval_rows`, loader keys) |
| `trainer` | `GRPOTrainerConfig` → TRL `GRPOConfig` (lr, batch, β, generations, vLLM, eval, Hub, `report_to`, `framework`) |
| `sft` | `SFTTrainerConfig` → TRL `SFTConfig` (`max_length` 2048) |
| `dpo` / `apo` | `DPOTrainerConfig` / `APOTrainerConfig` → TRL `DPOConfig` (`loss_type`, `max_length`) |
| `distillation` | `enabled`, `backend`, `model`, `tau` (0.5), `hitl`, Vertex / provider keys; see [Distillation](distillation.md) |
| `reward` | Weights for R_format, R_morph, R_rule, R_lang |
| `safety` | `SafetyConfig` → `SafetySpec` |
| `experiment` | `kveritas`, `kveritas_seal`, `kveritas_disclosure`, `kveritas_local`, `kveritas_local_key`, `freeze_resources`, `dataset`, `scope`, `preset`, `max_eval_rows`, `eval_source`, `eval_max_new_tokens` |
| `wordfreq` | `raw_inputs` (falls back to `data.source`, then packaged `beni/data/raw`) |

`distillation.provider` is a deprecated alias of `distillation.backend`.

Env keys: `SEBENI_HOME`, `SEBENI_WORKING_DIR`, `SEBENI_ENV` (extra `.env` file),
`GOOGLE_API_KEY` (alias `GOOGLE_API`), `GOOGLE_APPLICATION_CREDENTIALS`,
`GOOGLE_CLOUD_PROJECT` / `GOOGLE_PROJECT_ID` / `GOOGLE_PROJECT`,
`GOOGLE_CLOUD_LOCATION` / `GOOGLE_LOCATION` / `GOOGLE_REGION`,
`OPENAI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `HF_TOKEN`.
`.env` files are read from the cwd, the config file directory, and the working
directory; they never override variables already set.

Extras: `[train]`, `[wandb]`, `[distil]`, `[gguf]`, `[jax]`, `[docs]`, `[dev]`,
`[gui]` (`torchao` and `bitsandbytes` are in `[train]`). Install the headless parser separately with
`pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps`;
`[gui]` is wxPython for upstream gparser only.

## Python equivalent of `sebeni train`

```python
from beni.core.srl.config import MasterConfig
from beni.core.srl.unified import SRLTrainer

cfg = MasterConfig.from_yaml("config.yaml")
trainer = SRLTrainer(cfg)
trainer.train([{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}])
```
