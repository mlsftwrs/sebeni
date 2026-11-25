# Matrix and cloud runs

The `scripts/` directory holds the research tooling used for the multi-model
experiment matrix: batch runners, Hub publishing, re-scoring, zero-shot checks,
and Google Cloud launchers. None of it is part of the `sebeni` console script.
Run it from a **repository clone** (the scripts import `beni` from the repo root
and read `configs/matrix/`).

```bash
git clone https://github.com/mlsftwrs/sebeni.git && cd sebeni
pip install -e ".[train,distil]"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
python scripts/run_matrix.py --models SmolLM2-135M --algorithms sft --max-steps 2 \
  --backend algorithmic --max-eval-rows 5 --dry-run
```

## Matrix configuration

| File | What |
| --- | --- |
| `configs/matrix/base_mali13.yaml` | Default base MasterConfig for every cell (read only): MALI13, 500 steps, bf16, vLLM on, Google distillation (`gemini-3.7-flash`, Vertex, `global`), `kveritas` + `kveritas_seal`, reward weights 0.2/0.3/0.3/0.2 |
| `configs/matrix/models.json` | The 12 candidate models (`id`, `slug`, `display_name`, `params`, `trust_remote_code`, `recommended_batch_size`) |

Top-level `preset:` / `scope:` map onto `experiment.preset` /
`experiment.scope`; other unknown keys log a warning ("Ignoring unknown config
key …"). Its `max_eval_rows: null` scores every held-out row per language;
pass `--max-eval-rows` for a quicker cap.

## `scripts/run_matrix.py`

Runs models × arms. Each cell builds a config from the base config
(`--base-config`, never modified), writes it to
`{output-dir}/{run_slug}/config.yaml`, and runs it as its own `sebeni exp`
subprocess. With K-Veritas on, that one monitored run covers distill → train →
eval and seals `{run_slug}/exp/report.pdf`, which is copied (with its
`.kvbundle.zip`) to `{run_slug}/kveritas_report.pdf`. It writes the "Section 10"
reporting frame as `matrix_results.json`, `matrix_results.csv`, and
`matrix_results.md` in `--output-dir`. GRPO cells get a batch size and
gradient accumulation that divide `num_generations`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--models` | all 12 in `models.json` | Comma-separated slugs or HF ids |
| `--algorithms` | `sft,grpo,dpo,apo` | Arms |
| `--scope` | `MALI13` | Scope |
| `--max-steps` | `500` | Steps per cell |
| `--load-in-4bit` / `--load-in-8bit` / `--no-quantization` | (YAML) | Mutually exclusive quantization |
| `--distil-model` | `gemini-3.7-flash` | Distiller teacher |
| `--backend` | `google` | `google`, `gemini`, or `algorithmic` |
| `--vertex` | on | Vertex AI client (cannot be turned off from the CLI) |
| `--location` | `global` | Also exported as `GOOGLE_CLOUD_LOCATION` / `GOOGLE_LOCATION` |
| `--base-config` | `configs/matrix/base_mali13.yaml` | Base MasterConfig (read only) |
| `--output-dir` | `experiments/matrix` | Outputs and reports |
| `--max-eval-rows` | (YAML: `null` = all) | Paragraphs scored per language |
| `--kveritas-disclosure` | `open` | K-Veritas disclosure mode |
| `--kveritas-local-key` | — | Sign reports with this local RSA key (PEM) instead of the attestation server (`experiment.kveritas_local`) |
| `--use-vllm` / `--no-use-vllm` | (YAML: on) | GRPO vLLM rollouts |
| `--resume` | off | Skip cells that already have a readable `exp/eval.json` |
| `--dry-run` | off | Write configs and manifests only |

## `scripts/run_and_push_matrix.py`

Runs all four arms for one model and pushes them into **one** Hugging Face repo
(one subdirectory per algorithm). It checks HF credentials first, creates or
verifies the repo (private by default), retrains every cell, writes the Section
10 frames and K-Veritas seals, and uploads weights, tokenizers, configs,
Trackio logs, and tables, replacing any existing algorithm subdirectory.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--hf-repo` | **required** | Target repo id |
| `--hf-token` | `HF_TOKEN` / CLI login | Token |
| `--public` | off | Public repo |
| `--models` | `SmolLM2-135M` | Model slug or id |
| `--algorithms` | `sft,grpo,dpo,apo` | Arms |
| `--max-steps` | `500` | Steps per arm |
| `--max-eval-rows` | `10` | Held-out and in-training eval rows |
| `--eval-steps` | `max_steps // 10` | Eval interval |
| `--max-completion-length` | `256` | Rollout length |
| `--load-in-4bit` | off | 4-bit (recommended on L4) |
| `--scope` / `--output-dir` | `MALI13` / `experiments/matrix` | |
| `--skip-distillation` / `--distill` | skip | Train on the packaged data without Gemini distillation |
| `--distil-model`, `--backend`, `--location`, `--vertex/--no-vertex` | `gemini-3.7-flash`, `google`, `global`, off | Distillation, when `--distill` |
| `--kveritas-disclosure` | `open` | |
| `--force-rerun` / `--no-force-rerun` | on | Overwrite existing cells |
| `--resume`, `--skip-completed` | off | Keep finished cells (needs `--no-force-rerun`) |
| `--restart-grpo` / `--no-restart-grpo` | on | Always retrain GRPO |
| `--use-vllm` / `--no-use-vllm` | on | GRPO vLLM rollouts |
| `--vllm-gpu-memory-utilization` | `0.3` | |
| `--dry-run` | off | Build configs only |

## `scripts/run_and_push_per_model.py`

N models × N arms, **one repo per cell**, named by `--repo-template`
(`{org}/sebeni-{model}-{algo}-{steps}`, for example
`mlsftwrs/sebeni-smollm2-135-sft-50`). Pushes weights, adapter configs,
tokenizers, training data, telemetry, configs, eval reports, and K-Veritas
seals, ready for `transformers` / `peft` inference.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--org` | `mlsftwrs` | HF organization |
| `--repo-template` | `{org}/sebeni-{model}-{algo}-{steps}` | Repo naming |
| `--hf-token`, `--public` | env, off | |
| `--include-weights` / `--no-weights` | include | Upload weights and tokenizers |
| `--models` | `SmolLM2-135M,LFM2.5-230M,Granite-4.0h-350M,Qwen2.5-0.5B,Gemma-4-E2B` | Or `all` |
| `--algorithms` | `sft,grpo,dpo,apo` | |
| `--max-steps` | `50` | |
| `--max-eval-rows` | `10` | |
| `--eval-steps`, `--max-completion-length`, `--load-in-4bit` | `max_steps // 10`, `256`, off | |
| `--skip-distillation` / `--distill` | skip | |
| `--distil-model`, `--backend`, `--location`, `--vertex` | `gemini-2.5-flash`, `google`, `global`, off | |
| `--force-rerun`, `--restart-grpo`, `--use-vllm` | on | Same as above |
| `--resume`, `--skip-completed` | off | |
| `--recompute-completed` / `--no-recompute-completed` | off | Only re-score already uploaded models |
| `--eval-only` | off | Skip training; evaluate aligned weights and update reports |
| `--vllm-gpu-memory-utilization` | `0.3` | |
| `--dry-run` | off | Print plan and repo names |

## `scripts/recompute_and_push_eval.py`

Re-scores trained weights (local run directory or HF repo) on held-out
`beni/data/test.json` and updates `README.md` and `eval_results.json` on the
Hub without retraining.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--repo-id` | — | One specific repo |
| `--models` | `SmolLM2-135M` | Or `all` |
| `--algorithms` | `grpo,sft` | |
| `--steps` | `100` | Step suffix used in repo names |
| `--scope` | `MALI13` | |
| `--output-dir` | `experiments/matrix` | Local runs root |
| `--hf-org`, `--hf-token` | `mlsftwrs`, env | |
| `--max-eval-rows` | `10` | Paragraphs per language |
| `--push-to-hub` / `--no-push` | push | |

## `scripts/zeroshot_10.py`

Zero-shot check of the five reported base models on
`beni/data/raw/dataset_10_samples.jsonl`. It applies the training remap
(`seq`→`spp`, `mlq`→`kao`, `hsy`→`mey`) and drops `bbo` unless
`--include-bbo`. Outputs go to `runs/zeroshot10/`.

```bash
python scripts/zeroshot_10.py slice                         # capped jsonl
python scripts/zeroshot_10.py generate --model smollm2-135m # greedy decode (GPU for large models)
python scripts/zeroshot_10.py score --model smollm2-135m    # format rate, then Φ / MER / MCS / UWEC on formatted rows
python scripts/zeroshot_10.py export-ls                     # Label Studio tasks + label_config.xml
```

| Flag | Default |
| --- | --- |
| `--data` | `beni/data/raw/dataset_10_samples.jsonl` |
| `--out` | `runs/zeroshot10` |
| `--per-language` | `2` |
| `--include-bbo` | off |
| `--model` | all five |
| `--max-new-tokens` | `512` |

Gemma-4-E2B is gated: set `HF_TOKEN` first. The notebook
[`cookbooks/eval_zeroshot.ipynb`](https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/eval_zeroshot.ipynb)
runs the same evaluation in Colab / Kaggle.

## Google Cloud

All launchers read `GOOGLE_CLOUD_PROJECT` (or the active `gcloud` project) and
most read `GOOGLE_CLOUD_REGION` (default `us-central1`).

| File | What |
| --- | --- |
| `Dockerfile.matrix` | Matrix image: `pip install -e ".[train,distil]"`, `GOOGLE_CLOUD_LOCATION=global`, entrypoint `scripts/run_matrix.py` (default command: a 2-step SmolLM2 GRPO trial) |
| `cloudbuild.yaml` | Cloud Build of `Dockerfile.matrix` into `$_IMAGE_NAME` |
| `scripts/deploy_cloud_run.sh` | Builds the image with Cloud Build (`cloudbuild.yaml`) into Artifact Registry, then creates or updates the Cloud Run job `sebeni-matrix-trial` on one L4 GPU |
| `scripts/run_vertex_l4.sh` | Submits the image as a Vertex AI custom job on `g2-standard-4` (1× L4) |
| `scripts/run_vertex_cpu.sh` | Same on `e2-standard-16` (CPU) |
| `scripts/run_gce_l4.sh [zone]` | Creates a GCE `g2-standard-4` L4 VM (`sebeni-l4-runner`), trying several zones; uses `scripts/startup_l4.sh` |
| `scripts/startup_l4.sh` | VM startup: NVIDIA driver, Docker, NVIDIA container toolkit, then the matrix container |
| `scripts/create_vertex_notebook_l4.sh` | Creates a Vertex AI Workbench instance with one L4 (`sebeni-l4-workbench`) |
| `scripts/run_workbench_cli.sh [model] [algo] [steps] [eval-rows]` | Workbench runner: installs `kveritas` if missing, writes an L4-tuned copy of the base config, then calls `run_matrix.py --base-config` with it |
| `scripts/run_workbench_cli.py` | Python version: `--model` (`SmolLM2-135M`), `--algo` (`grpo`), `--steps` (`2`), `--eval-rows` (`5`), `--project`, `--location` (`global`), `--output-dir` (`experiments/matrix_workbench`) |

Both workbench runners write their L4 settings (bf16 on, fp16 off, gradient
checkpointing on, 4-bit on) to `<output-dir>/base_mali13.l4.yaml` and pass
it as `--base-config`; `configs/matrix/base_mali13.yaml` is never modified.

The notebook
[`cookbooks/experimentation_matrix.ipynb`](https://github.com/mlsftwrs/sebeni/blob/main/cookbooks/experimentation_matrix.ipynb)
walks through the same matrix with Vertex AI credentials.
