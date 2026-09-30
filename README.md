# Sebeni: self-aware morphotactic generation for extremely low-resource languages

<p align="center">
  <img src="logo_rm.jpeg" alt="RobotsMali" width="220"/>
</p>

[![docs](https://img.shields.io/badge/docs-Documentation-indigo)](https://mlsftwrs.github.io/sebeni)
[![site](https://img.shields.io/badge/home-seben.robotsmali.org-blue)](https://seben.robotsmali.org)
[![hub](https://img.shields.io/badge/hub-mlsftwrs-yellow)](https://huggingface.co/mlsftwrs)
[![github](https://img.shields.io/badge/code-mlsftwrs/sebeni-black)](https://github.com/mlsftwrs/sebeni)

Sebeni is a morphotactic post-training toolkit for Manding and related
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
| `[train]` | torch, transformers, trl, peft, datasets, accelerate, trackio |
| `[wandb]` | wandb (`trainer.report_to: wandb`) |
| `[distil]` | google-genai, openai, groq, together |
| `[gguf]` | llama-cpp-python for local Distiller refinement |
| `[jax]` | JAX, Flax, Optax, Orbax policy updates |
| `[docs]` | MkDocs Material + mkdocstrings |
| `[dev]` | pytest, ruff |
| `[gui]` | wxPython (upstream Daba gparser only; not required) |

The default algorithmic Distiller needs no key. Optional LLM refinement accepts
`.env`, Google ADC, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`,
or `TOGETHER_API_KEY`. Hub push uses `HF_TOKEN` or `huggingface-cli login`.

## Quick start

**Canonical experiment** (packaged jsonl, no data to write):

```bash
sebeni exp --preset multi13 --algorithm sft -w ./runs/multi13-sft
sebeni exp --preset single --lang bam --algorithm grpo -w ./runs/bam-grpo
```

If `-c` and `--preset` are omitted, Sebeni loads the MULTI13 preset.
`sebeni exp` trains on `beni/data/raw/dataset_300_samples.jsonl` and
evaluates `beni/data/test.json`. Details:
[Experiments](https://mlsftwrs.github.io/sebeni/experiments/).

**Your own jsonl:**

```bash
sebeni init --lang bam --lang mku -w ./runs/manding-001
# point data.source at jsonl/csv with text + lang
sebeni distill -c ./runs/manding-001/config.yaml
sebeni train   -c ./runs/manding-001/config.yaml --algorithm grpo --lr 1e-5 --max-steps 50 --kveritas --eval-ratio 0.1 --eval-steps 10
sebeni eval    -c ./runs/manding-001/config.yaml
sebeni generate -c ./runs/manding-001/config.yaml --prompt "Aw ka kɛnɛ wa?" --lang bam
sebeni wordfreq -c ./runs/manding-001/config.yaml
```

`init` writes `config.yaml` plus `data/`, `models/`, `runs/`, `exp/`, `runtime/`.
`--lang multi13` (or `all`) is the 13 canonical groups. `--lang bam` is one
language. Trainer / LoRA knobs are YAML keys under `model:` / `trainer:` /
`dpo:` / `apo:` and CLI flags on `sebeni train`. Full tables:
[Hyperparameters](https://mlsftwrs.github.io/sebeni/hyperparams/).

Presets: [`configs/presets/multi13.yaml`](configs/presets/multi13.yaml),
[`configs/presets/single.yaml`](configs/presets/single.yaml).
`configs/exp.yaml` is an alias of the MULTI13 preset.

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
   Freeze G and D under `{working_dir}/data/resources/{lang}/`.
2. **Train one arm.** SFT, GRPO, DPO, or APO reads that checkpoint. Arms do not
   write a new grammar or dictionary.
3. **Evaluate.** Held-out `test.json` reports MER, MCS, and UWEC for model,
   algorithm, and scope. All three are costs to minimize. MULTI13 pools
   morphemes (MER) and tokens (MCS, UWEC). UWEC is evaluation-only.

One policy θ; `(G_ℓ, D_ℓ)` per language. τ defaults to **0.5**. Full narrative:
[SAMPG](https://mlsftwrs.github.io/sebeni/sampg/). Metrics:
[Rewards](https://mlsftwrs.github.io/sebeni/rewards/).

## CLI

```
sebeni init      --lang bam --lang mku -w ./runs/manding-001
sebeni distill   -c config.yaml
sebeni train     -c config.yaml --algorithm grpo [--lr 1e-5] [--kveritas] [--eval-ratio 0.1]
sebeni eval      -c config.yaml
sebeni exp       --preset multi13 --algorithm sft -w ./runs/multi13-sft
sebeni wordfreq  -c config.yaml
sebeni generate  -c config.yaml --prompt "..." --lang bam
sebeni push      -c config.yaml --repo-id mlsftwrs/<model>
```

Working directory, later wins if set: `~/.sebeni` → `SEBENI_HOME` /
`SEBENI_WORKING_DIR` → YAML `working_dir:` → CLI `-w`.

| Artifact | Path |
| --- | --- |
| Frozen G, D | `{working_dir}/data/resources/{lang}/` |
| G, D checkpoints | `{working_dir}/data/baselines/{lang}/baseline.gram` `.dict` and `baseline_vN` |
| Policy + tokenizer | `{working_dir}/models/` |
| Hub card + snapshot | `{working_dir}/models/README.md`, `safety_snapshot.json` |
| Eval / wordfreq | `{working_dir}/exp/eval.json`, `{working_dir}/exp/wordfreq/` |
| mparser runtime | `{working_dir}/runtime/` |

If packaged `beni/data/baselines/{lang}/` is missing, Distiller writes Daba-compatible
stubs and uses **bootstrap** prompts. First promote is a parseable-file gate;
later checkpoints require Φ′ > Φ.

## Safety / Hub

`SafetyGovernor` is consulted on every G/D promote, policy update, and Hub
export:

- Completions must be valid JSON with `tokens`; format-invalid batches cannot update θ
- `R_lang` vs **that row**'s group code
- Promote G, D only when Φ′ > Φ (after the scratch parse gate)
- Training distrust U and KL-to-ref (`beta`) down-weight noisy rewards
- **No Hub push** without a model card and `safety_snapshot.json` (Φ, τ, checkpoint id)

Push checklist and org transfer: [Hub](https://mlsftwrs.github.io/sebeni/hub/).

## Cookbooks & Evaluation Notebooks

Ready-to-run Jupyter notebooks are provided in [`cookbooks/`](cookbooks/):

| Notebook | Purpose |
| --- | --- |
| [`cookbooks/eval_grpo.ipynb`](cookbooks/eval_grpo.ipynb) | End-to-end GRPO training and held-out evaluation on `dataset_300_samples.jsonl` with K-Veritas stream |
| [`cookbooks/eval_dpo.ipynb`](cookbooks/eval_dpo.ipynb) | End-to-end DPO training and evaluation on `dataset_300_samples.jsonl` |
| [`cookbooks/eval_apo.ipynb`](cookbooks/eval_apo.ipynb) | End-to-end APO training and evaluation on `dataset_300_samples.jsonl` |
| [`cookbooks/eval_sft.ipynb`](cookbooks/eval_sft.ipynb) | End-to-end SFT training and evaluation on `dataset_300_samples.jsonl` |
| [`cookbooks/run_exp.ipynb`](cookbooks/run_exp.ipynb) | Rapid multi-arm benchmark (`sebeni exp`) across all algorithms on `dataset_10_samples.jsonl` |

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
- Documentation: [seben.robotsmali.org/docs](https://mlsftwrs.github.io/sebeni)
- Project home: [seben.robotsmali.org](https://seben.robotsmali.org)
- Organization: [RobotsMali](https://robotsmali.org)

## License

MIT (this tree). Daba remains GPLv2+.

Sebeni - write in Malian languages.  
[seben.robotsmali.org](https://seben.robotsmali.org)·
[seben.robotsmali.org/docs](https://seben.robotsmali.org/docs)
