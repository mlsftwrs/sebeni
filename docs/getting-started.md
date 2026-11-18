# Getting started

Install Sebeni, write `{text, lang}` rows (or use packaged data), then
`distill` → `train` → `eval`. Distill once, freeze G and D, train one arm,
score held-out MER / MCS / UWEC. Zero data prep:
[Experiments](experiments.md) (`sebeni exp`). Copy-paste recipes:
[Use cases](use-cases.md).

Public docs: [https://mlsftwrs.github.io/sebeni](https://mlsftwrs.github.io/sebeni).

## Install

Python ≥ 3.10. Primary install is GitHub (not an editable checkout):

```bash
pip install "sebeni[train,distil] @ git+https://github.com/mlsftwrs/sebeni.git"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
sebeni --help
```

| Extra | What |
| --- | --- |
| (default) | CLI, YAML, DabaX runtime dependencies, metrics, safety |
| `[train]` | torch, transformers, trl, peft, datasets, accelerate, vllm, trackio, `torchao`, `bitsandbytes` |
| `[wandb]` | wandb (for `trainer.report_to: wandb`) |
| `[distil]` | google-genai, openai, groq, together |
| `[gguf]` | llama-cpp-python for local Distiller refinement |
| `[jax]` | JAX, Flax, Optax, Orbax policy updates |
| `[docs]` | MkDocs Material + mkdocstrings, ruff |
| `[dev]` | pytest, pytest-mock, ruff, datasets |
| `[gui]` | wxPython — **not** required |

`vllm` is used only by GRPO with `trainer.use_vllm: true` on CUDA. Without
CUDA or vllm, GRPO falls back to Hugging Face generation.

On Linux, `pip install` also tries to download the optional
[K-Veritas](https://kveritas.org/docs) binary when `kveritas` is not on `PATH`.
See [Experiments → K-Veritas](experiments.md#k-veritas-optional).

Parser credit: [maslinych/daba](https://github.com/maslinych/daba) (GPLv2+).
Install it from GitHub with `--no-deps`; DabaX does **not** require wxPython.
Sebeni already pins the CLI runtime (`setuptools>=65,<81` / `pkg_resources`,
`funcparserlib`, `intervaltree`, `pytrie`, `attrdict3`, `regex`).

From a clone:

```bash
pip install -e ".[train,distil,dev]"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
```

The default algorithmic Distiller needs no key. Optional LLM backends read
`.env`, Google ADC, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`,
`TOGETHER_API_KEY`, `HF_TOKEN`. Skip Distiller with `distillation.enabled: false`,
or run `sebeni train` after a prior `sebeni distill`. Policy arms never promote G or D.

## Minimal dataset

Every row is one sentence and its language. G and D are **files**, not columns.

```json
{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}
{"text": "i ni ce", "lang": "mku"}
```

CSV (`text,lang`) and `.txt` (one sentence per line; lang from filename or
`default_lang`) also work. Hugging Face datasets: set `data.source` to the
id and `hf_lang_key` if the column is not `lang`. Preference scheme (DPO/APO):
`{text, lang, chosen, rejected}` or `{text, lang, completions, scores}`.

## Happy path

```mermaid
flowchart LR
  Install --> Init["sebeni init"]
  Init --> Distill["sebeni distill"]
  Distill --> Train["sebeni train"]
  Train --> Eval["sebeni eval"]
  Eval --> Next["generate / wordfreq / push"]
```

```bash
sebeni init --lang multi13 -w ./runs/multi13
sebeni distill -c ./runs/multi13/config.yaml
sebeni train -c ./runs/multi13/config.yaml --algorithm grpo --lr 1e-5 --max-steps 20
sebeni eval  -c ./runs/multi13/config.yaml
sebeni exp --preset single --lang bam --algorithm sft -w ./runs/bam-sft
```

`init` creates `config.yaml` plus `data/`, `models/`, `runs/`, `exp/`,
`runtime/` (default `-w ./runs/sebeni-001`). `--lang multi13` is the default
scope. `--lang bam` is one language; `--lang bam,mku` is a list.

!!! note
    `sebeni eval` scores the trained weights in `models/` (override with
    `--weights`) on every held-out row per language
    (`experiment.max_eval_rows: null`, the default). Without trained weights it
    warns and scores the base model.

Trainer / LoRA knobs: [Hyperparameters](hyperparams.md) or `sebeni train --help`.

## Cookbooks

Open a notebook in Colab or Kaggle. Each installs Sebeni from GitHub (no clone).

| Notebook | Launch |
| --- | --- |
| [eval_grpo.ipynb](https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_grpo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_grpo.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_grpo.ipynb) |
| [eval_dpo.ipynb](https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_dpo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_dpo.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_dpo.ipynb) |
| [eval_apo.ipynb](https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_apo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_apo.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_apo.ipynb) |
| [eval_sft.ipynb](https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_sft.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_sft.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_sft.ipynb) |
| [eval_zeroshot.ipynb](https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_zeroshot.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_zeroshot.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_zeroshot.ipynb) |
| [experimentation_matrix.ipynb](https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/experimentation_matrix.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/experimentation_matrix.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/experimentation_matrix.ipynb) |
| [run_exp.ipynb](https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/run_exp.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/run_exp.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/run_exp.ipynb) |

## Relocatable working directory

Later wins if set:

1. `~/.sebeni`
2. `SEBENI_HOME` / `SEBENI_WORKING_DIR`
3. YAML `working_dir:`
4. CLI `-w`

Relative YAML paths resolve against the config file directory; relative `-w`
against cwd. Tests should pass `-w` a temp dir.

## Scratch language

No packaged `beni/data/baselines/{lang}/` → Distiller writes stubs under
`{working_dir}/data/baselines/{lang}/` and uses bootstrap prompts. First
promote: parseable files. Later: Φ′ > Φ. Maninka group is **mku**.

## Artifacts

| Path | What |
| --- | --- |
| `{working_dir}/data/baselines/{lang}/` | `baseline.gram` / `.dict`, `baseline_vN` — the G, D every arm and eval parse with |
| `{working_dir}/data/resources/{lang}/` | Frozen snapshot: `baseline.*`, `distilled.*`, `hashes.json` |
| `{working_dir}/data/resources/frozen.json` | Freeze marker; `sebeni train` skips distillation when present |
| `{working_dir}/data/sft/train.jsonl` | SFT labels (plus `exp/sft_label_report.json`) |
| `{working_dir}/models/` | Policy, tokenizer, Hub `README.md`, `safety_snapshot.json` |
| `{working_dir}/exp/` | `eval.json` (Φ, MER, MCS, UWEC), `manifest.json`, `wordfreq/`, `report.pdf` / `report_train.pdf` / `report_eval.pdf` (K-Veritas seals of `exp` / `train` / `eval`) |
| `{working_dir}/runtime/` | Headless mparser |
