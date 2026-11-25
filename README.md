# Sebeni: self-aware morphotactics generation for extremely low-resource languages

<p align="center">
  <img src="logo_rm.jpeg" alt="RobotsMali" width="220"/>
</p>

[![docs](https://img.shields.io/badge/docs-Documentation-indigo)](https://mlsftwrs.github.io/sebeni)
[![site](https://img.shields.io/badge/home-seben.robotsmali.org-blue)](https://seben.robotsmali.org)
[![hub](https://img.shields.io/badge/hub-mlsftwrs-yellow)](https://huggingface.co/mlsftwrs)
[![github](https://img.shields.io/badge/code-mlsftwrs/sebeni-black)](https://github.com/mlsftwrs/sebeni)

Sebeni is a morphotactics post-training toolkit for Manding and related
extremely low-resource languages. One run is three stages: 

- **distill** grammar G and dictionary D once per language, 
- **train one arm** (SFT, GRPO, DPO, or APO) against those frozen files, then 
- **evaluate** held-out MER, MCS, and UWEC.

**Completions are morphological JSON, not a chatbot.**

```bash
pip install "sebeni[train,distil] @ git+https://github.com/mlsftwrs/sebeni.git"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
sebeni --help
```

- Docs: [mlsftwrs.github.io/sebeni](https://mlsftwrs.github.io/sebeni)
- Project home: [seben.robotsmali.org](https://seben.robotsmali.org)
- Hub org: [huggingface.co/mlsftwrs](https://huggingface.co/mlsftwrs)
- Contact: [seben@robotsmali.org](mailto:seben@robotsmali.org)

## Install

Python ≥ 3.10. The default extra does **not** install wxPython.

```bash
pip install "sebeni[train,distil] @ git+https://github.com/mlsftwrs/sebeni.git"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
```

From a clone, install Sebeni and then the same headless Daba parser:

```bash
pip install -e ".[train,distil,dev]"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
```

| Extra | Contents |
| --- | --- |
| (default) | CLI, YAML, DabaX runtime dependencies, metrics, safety |
| `[train]` | torch, transformers, trl, peft, datasets, accelerate, vllm, trackio, `torchao`, `bitsandbytes` |
| `[wandb]` | wandb (`trainer.report_to: wandb`) |
| `[distil]` | google-genai, openai, groq, together |
| `[gguf]` | llama-cpp-python for local Distiller refinement |
| `[jax]` | JAX, Flax, Optax, Orbax policy updates |
| `[docs]` | MkDocs Material + mkdocstrings, ruff |
| `[dev]` | pytest, pytest-mock, ruff, datasets |
| `[gui]` | wxPython (upstream Daba gparser only; not required) |

`vllm` is only used by GRPO with `trainer.use_vllm: true` on a CUDA machine;
without CUDA or vllm, GRPO falls back to Hugging Face generation.

On Linux, `pip install` also tries to download the
[K-Veritas](https://kveritas.org/docs) binary (from `27-GROUP/kveritas-releases`)
into the environment `bin/`, then `/usr/local/bin`, then `~/.local/bin`, when
`kveritas` is not already on `PATH`. It is optional; without it Sebeni warns
and runs unmonitored. With `--kveritas` (or `experiment.kveritas: true`),
`sebeni exp`, `train` and `eval` re-launch themselves once under `kveritas run`;
do not wrap them in `kveritas run` yourself. `experiment.kveritas_local: true`
signs with a local key instead of an attestation server.

The default algorithmic Distiller needs no key. Optional LLM refinement accepts
`.env`, Google ADC, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`,
or `TOGETHER_API_KEY`. Hub push uses `HF_TOKEN` or `huggingface-cli login`.

## Quick start

**Canonical experiment** (packaged jsonl, no data to write):

```bash
sebeni exp --preset mali13 --algorithm sft -w ./runs/mali13-sft
sebeni exp --preset single --lang bam --algorithm grpo -w ./runs/bam-grpo
```

If `-c` and `--preset` are omitted, Sebeni loads the MALI13 preset
(`beni/data/presets/mali13.yaml`, shipped in the package). `sebeni exp` trains
on `data.source` when the `-c` config sets it, otherwise on the packaged
`beni/data/raw/dataset_300_samples.jsonl`; `--preset` always uses the packaged
file. It evaluates `beni/data/test.json` unless `experiment.eval_source` is set.
Details: [Experiments](https://mlsftwrs.github.io/sebeni/experiments/).

**Your own jsonl:**

```bash
sebeni init --lang bam --lang mku -w ./runs/manding-001
# point data.source at jsonl/csv with text + lang
sebeni distill -c ./runs/manding-001/config.yaml
sebeni train   -c ./runs/manding-001/config.yaml --algorithm grpo --lr 1e-5 --max-steps 50 --kveritas --eval-ratio 0.1 --eval-steps 10
sebeni eval    -c ./runs/manding-001/config.yaml   # scores the trained weights in models/
sebeni generate -c ./runs/manding-001/config.yaml --prompt "Aw ka kɛnɛ wa?" --lang bam
sebeni wordfreq -c ./runs/manding-001/config.yaml
```

`init` writes `config.yaml` plus `data/`, `models/`, `runs/`, `exp/`, `runtime/`
(default `-w ./runs/sebeni-001`, default `--lang mali13`).
`--lang mali13` (or `all`) is the 13 canonical groups. `--lang bam` is one
language; `--lang bam,mku` is a list. Trainer / LoRA knobs are YAML keys under
`model:` / `trainer:` (GRPO) / `sft:` / `dpo:` / `apo:` and CLI flags on
`sebeni train`. Full tables:
[Hyperparameters](https://mlsftwrs.github.io/sebeni/hyperparams/).

`--eval-ratio` (default 0.1) or `--eval-rows` (a row count; 0 disables the
split) holds out part of the training rows for in-training eval
(`data.eval_ratio` / `data.eval_rows`). `experiment.max_eval_rows` only caps
`sebeni eval`; its default `null` scores every held-out row per language.

Presets: [`beni/data/presets/mali13.yaml`](beni/data/presets/mali13.yaml),
[`beni/data/presets/single.yaml`](beni/data/presets/single.yaml).
[`configs/exp.yaml`](configs/exp.yaml) is a separate MALI13 config (same as the
packaged `beni/data/exp.yaml`). It differs from the preset in reward weights:
0.1/0.4/0.4/0.1 vs the preset's 0.2/0.3/0.3/0.2 (format/morph/rule/lang).

## Dataset: text and language

Every element is one sentence plus its language identity — not a grammar, not
a dictionary:

```json
{"text": "aw ka ne labato.", "lang": "bam"}
{"text": "i ni ce", "lang": "mku"}
```

| Field | Meaning |
| --- | --- |
| `text` | Surface sentence (the string DabaX parses and the policy conditions on) |
| `lang` | ISO or Sebeni group code for **this** row (`language` is accepted as an alias) |

Load from JSONL/CSV/TXT, a directory, a list of paths, or a Hugging Face dataset
id (`data.source` in YAML). Unlabeled rows fall back to `data.default_lang`.
A mixed-language file is normal: one language per row. Completions must not mix
languages **inside** a single JSON object (`R_lang`). G and D are files
(`baseline.gram` / `baseline.dict`), not columns.

Maninka group code is **MKU** (not MLQ). In the experiment jsonl only, `mlq`
maps to Kassonke `kao`, `hsy` to `mey`, and `seq` to `spp`. `bbo` is an outlier
kept for `--lang bbo`. Completions use JSON `tokens`.

## How a run works

1. **Distill once.** Split the train rows by language. Score Φ with DabaX. If
   Φ < τ, Distiller proposes \(G_{cand}, D_{cand}\) and promotes only when Φ′ > Φ.
   The latest checkpoint lives in `{working_dir}/data/baselines/{lang}/`; a
   hashed snapshot is copied to `{working_dir}/data/resources/{lang}/` and
   `data/resources/frozen.json` marks the run as frozen. `sebeni train` skips
   distillation when that marker exists; `sebeni exp` always re-distills.
2. **Train one arm.** SFT, GRPO, DPO, or APO parses with the latest G, D in
   `data/baselines/{lang}/`. Arms do not write a new grammar or dictionary.
3. **Evaluate.** Held-out packaged `test.json` (or `experiment.eval_source` /
   `sebeni eval --eval-file`) reports Φ, MER, MCS, and UWEC for
   model, algorithm, and scope. **MER and UWEC are costs to minimize; MCS is a
   consistency score (match fraction), higher is better.** MALI13 pools
   morphemes (MER) and tokens (MCS, UWEC); Φ is a macro mean over languages.
   UWEC is evaluation-only. `sebeni eval` decodes greedily with the trained
   weights (`--weights`, default `{working_dir}/models`) and the same prompt as
   training; without trained weights it warns and scores the base model.

One policy θ; `(G_ℓ, D_ℓ)` per language. τ defaults to **0.5**. Full narrative:
[SAMPG](https://mlsftwrs.github.io/sebeni/sampg/). Metrics:
[Rewards](https://mlsftwrs.github.io/sebeni/rewards/).

## CLI

```
sebeni init      --lang bam --lang mku -w ./runs/manding-001
sebeni distill   -c config.yaml
sebeni train     -c config.yaml --algorithm grpo [--lr 1e-5] [--kveritas] [--eval-ratio 0.1] [--use-vllm]
sebeni eval      -c config.yaml [--weights PATH|REPO] [--eval-file PATH]
sebeni exp       --preset mali13 --algorithm sft -w ./runs/mali13-sft   # alias: sebeni experiment
sebeni wordfreq  -c config.yaml
sebeni generate  -c config.yaml --prompt "..." --lang bam [--max-new-tokens 512]
sebeni push      -c config.yaml --repo-id mlsftwrs/<model> [--write-card]
```

Working directory, later wins if set: `~/.sebeni` → `SEBENI_HOME` /
`SEBENI_WORKING_DIR` → YAML `working_dir:` → CLI `-w`.

| Artifact | Path |
| --- | --- |
| G, D checkpoints (read by arms and eval) | `{working_dir}/data/baselines/{lang}/baseline.gram` `.dict` and `baseline_vN` |
| Frozen snapshot | `{working_dir}/data/resources/{lang}/` (`baseline.*`, `distilled.*`, `hashes.json`) |
| Freeze marker | `{working_dir}/data/resources/frozen.json` |
| SFT labels | `{working_dir}/data/sft/train.jsonl`, `{working_dir}/exp/sft_label_report.json` |
| Policy + tokenizer | `{working_dir}/models/` |
| Hub card + snapshot | `{working_dir}/models/README.md`, `safety_snapshot.json` (next to the evaluated weights; base-model evals write to `exp/`) |
| Eval | `{working_dir}/exp/eval.json`, `{working_dir}/exp/manifest.json` |
| Wordfreq | `{working_dir}/exp/wordfreq/wordfreq.json`, `{working_dir}/exp/wordfreq/{lang}/` |
| K-Veritas seal | `{working_dir}/exp/report.pdf` (`exp`), `exp/report_train.pdf` (`train`), `exp/report_eval.pdf` (`eval`) (`experiment.kveritas_seal: true`) |
| mparser runtime | `{working_dir}/runtime/` |

If packaged `beni/data/baselines/{lang}/` is missing, Distiller writes Daba-compatible
stubs and uses **bootstrap** prompts. First promote is a parseable-file gate;
later checkpoints require Φ′ > Φ.

## Safety / Hub

`SafetyGovernor` is consulted on G/D promote, policy update, and Hub export.

Enforced:

- Promote G, D only when Φ′ > Φ (after the scratch parse gate), candidate MER
  is within `safety.max_mer`, and the dictionary's `\lang` header matches
- All four arms (SFT, GRPO, DPO, APO) feed the governor. GRPO judges its own
  samples: a batch is refused only when no completion is good (all
  format-invalid, none in the right language, or all mixed-language). SFT, DPO
  and APO judge their training targets: any invalid, wrong-language or
  mixed-language target refuses the step, and such rows are dropped before
  training. Uncertainty U and the KL bound (`safety.max_kl`) use per-token
  log-ratios
- `require_language` and `refuse_mixed_language` are enforced; a refused update
  skips `optimizer.step` entirely (θ unchanged) and is counted in `blocked_steps`
- `sebeni push` needs `README.md` and a `safety_snapshot.json` written by
  `sebeni eval` for exactly the pushed weight files; nothing is auto-written
  (`push --write-card` re-renders the card from the eval snapshot)

Push checklist and org transfer: [Hub](https://mlsftwrs.github.io/sebeni/hub/).

## Cookbooks & Evaluation Notebooks

Ready-to-run Jupyter notebooks are in [`cookbooks/`](cookbooks/). Open them in Colab or Kaggle — no repo clone is required; each notebook installs Sebeni from GitHub.

| Notebook | Purpose | Launch |
| --- | --- | --- |
| [`cookbooks/eval_grpo.ipynb`](cookbooks/eval_grpo.ipynb) | End-to-end GRPO training and held-out evaluation on `dataset_300_samples.jsonl` with K-Veritas stream | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_grpo.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_grpo.ipynb) |
| [`cookbooks/eval_dpo.ipynb`](cookbooks/eval_dpo.ipynb) | End-to-end DPO training and evaluation on `dataset_300_samples.jsonl` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_dpo.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_dpo.ipynb) |
| [`cookbooks/eval_apo.ipynb`](cookbooks/eval_apo.ipynb) | End-to-end Anchored Preference Optimization (APO, [paper](https://huggingface.co/papers/2408.06266)) training and evaluation on `dataset_300_samples.jsonl` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_apo.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_apo.ipynb) |
| [`cookbooks/eval_sft.ipynb`](cookbooks/eval_sft.ipynb) | End-to-end SFT training and evaluation on `dataset_300_samples.jsonl` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_sft.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_sft.ipynb) |
| [`cookbooks/eval_zeroshot.ipynb`](cookbooks/eval_zeroshot.ipynb) | Zero-shot morphological evaluation of the five reported base models (no training) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/eval_zeroshot.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_zeroshot.ipynb) |
| [`cookbooks/experimentation_matrix.ipynb`](cookbooks/experimentation_matrix.ipynb) | Multi-model × multi-arm matrix with K-Veritas audit (Vertex AI distillation; see [Matrix and cloud runs](https://mlsftwrs.github.io/sebeni/scripts/)) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/experimentation_matrix.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/experimentation_matrix.ipynb) |
| [`cookbooks/run_exp.ipynb`](cookbooks/run_exp.ipynb) | Rapid multi-arm benchmark (`sebeni exp`) across all algorithms, on the `data.source` the notebook sets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlsftwrs/sebeni/blob/main/cookbooks/run_exp.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/run_exp.ipynb) |

## Parser

DabaX uses CLI `daba.mparser` only (`DictLoader`, `GrammarLoader`, `Tokenizer`,
`Processor`). Parser credit: [maslinych/daba](https://github.com/maslinych/daba)
(GPLv2+). Install it from GitHub with `--no-deps` to avoid its optional GUI
stack; Sebeni already pins `setuptools` (`pkg_resources`) and the other CLI
runtime libraries. Do not vendor GPL sources into this MIT tree. A CLI-only fork under
[mlsftwrs](https://github.com/mlsftwrs) is the intended long-term pin.

## Tests

```bash
pytest tests
```

## Contact & Support

- Contact email: [seben@robotsmali.org](mailto:seben@robotsmali.org)
- Documentation: [mlsftwrs.github.io/sebeni](https://mlsftwrs.github.io/sebeni)
- Project home: [seben.robotsmali.org](https://seben.robotsmali.org)
- Organization: [RobotsMali](https://robotsmali.org)

## License

MIT (this tree). Daba remains GPLv2+.

Sebeni - write in Malian languages.  
[seben.robotsmali.org](https://seben.robotsmali.org)·
[mlsftwrs.github.io/sebeni](https://mlsftwrs.github.io/sebeni)
