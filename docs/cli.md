# CLI and config

Public docs: [https://seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).

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

YAML overlays `MasterConfig` dataclasses (no second config system). Presets:
`configs/presets/multi13.yaml`, `configs/presets/single.yaml`.
`configs/exp.yaml` is an alias of the MULTI13 preset. Full trainer / LoRA tables:
[Hyperparameters](hyperparams.md).

## Commands

| Command | Purpose | Important flags |
| --- | --- | --- |
| `sebeni init` | Write `config.yaml` + workdir (`data/`, `models/`, `runs/`, `exp/`, `runtime/`) | `--lang multi13` or one code, `-w` |
| `sebeni distill` | Upstream SAMPG once per language; freeze G and D | `-c`, `-w`, `--lang`, `--hitl` |
| `sebeni train` | One arm against frozen G and D | `-c`, `--algorithm`, `-w`, `--lang`, `--lr`, `--max-steps` |
| `sebeni exp` | Distill, one arm, held-out eval (`experiment` alias) | `--preset multi13\|single`, `--algorithm`, `--lang`, `-c`, `-w` |
| `sebeni eval` | Held-out MER / MCS / UWEC → `{working_dir}/exp/eval.json` | `-c`, `-w`, `--lang` |
| `sebeni wordfreq` | Surfaces / lemmas / morphemes / stages per language | `-c`, `-w`, `--lang` |
| `sebeni generate` | Decode from saved policy; warn on `R_format` / `R_lang` | `-c`, `--prompt`, `--lang`, `--max-length` |
| `sebeni push` | Hub upload (card + snapshot required) | `-c`, `--repo-id` |

`--lang multi13` or `--lang all` selects the 13 canonical group codes.
`--lang bam` is one language. `--lang bam,mku` is a comma-separated list.
`--lang bbo` is the outlier and is not mixed into MULTI13.

Setting `--epochs` without `--max-steps` sets `max_steps: -1` so Hugging Face
runs by epoch count.

## `eval.json` shape

```json
{
  "model": "Qwen/Qwen2.5-0.5B",
  "algorithm": "sft",
  "scope": "MULTI13",
  "phi": 0.63,
  "mer": 0.21,
  "mcs": 0.18,
  "uwec": 1.04,
  "n_morphemes": 1840,
  "n_tokens": 920,
  "beta": 0.1,
  "eps": 1e-8,
  "reference_model": "Qwen/Qwen2.5-0.5B",
  "languages": ["bam", "mku"],
  "by_language": {
    "bam": {
      "phi": 0.71,
      "mer": 0.19,
      "mcs": 0.16,
      "uwec": 1.02,
      "n_morphemes": 1200,
      "n_tokens": 600,
      "checkpoint_id": "baseline_v3",
      "language": "bam",
      "scope": "MULTI13"
    }
  }
}
```

## MasterConfig fields

| Field | Meaning |
| --- | --- |
| `project_name` | Trackio / run name |
| `algorithm` | `sft` / `grpo` (default) / `dpo` / `apo` |
| `working_dir` | Relocatable root |
| `model` | `ModelConfig` (base, ref, LoRA rank/alpha/dropout, 4-bit) |
| `data` | `DataConfig` (`source`, `scheme`, `languages`, `default_lang`) |
| `trainer` | `GRPOTrainerConfig` → TRL `GRPOConfig` (lr, batch, β, generations, Hub, `report_to`) |
| `dpo` / `apo` | Policy-update sibling configs (same optimizer fields) |
| `distillation` | `enabled`, `provider`, `model`, `tau` (0.5), `hitl` |
| `reward` | Weights for R_format, R_morph, R_rule, R_lang |
| `safety` | `SafetyConfig` → `SafetySpec` |
| `experiment` | `kveritas` / `kveritas_seal` for `sebeni exp` |

Env keys: `SEBENI_HOME`, `SEBENI_WORKING_DIR`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`,
`GROQ_API_KEY`, `TOGETHER_API_KEY`, `HF_TOKEN`.

Extras: `[train]`, `[wandb]`, `[distil]`, `[docs]`, `[dev]`. Install the
headless parser separately with
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
