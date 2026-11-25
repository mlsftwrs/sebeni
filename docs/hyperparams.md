# Model and trainer hyperparameters

Every field on `ModelConfig`, `GRPOTrainerConfig`, `SFTTrainerConfig`,
`DPOTrainerConfig`, and `APOTrainerConfig` is a YAML key under `model:`,
`trainer:` (GRPO), `sft:`, `dpo:`, or `apo:`. `sebeni train` overlays the same
knobs from the CLI (non-None flags win). `algorithm:` picks which block the
shared flags land in.

Docs site: [mlsftwrs.github.io/sebeni](https://mlsftwrs.github.io/sebeni).

```bash
sebeni train -c config.yaml \
  --lr 1e-5 --batch-size 4 --grad-accum 4 --max-steps 100 \
  --beta 0.04 --num-generations 8 --temperature 0.8 \
  --lora-r 32 --lora-alpha 64 --bf16 --grad-checkpoint \
  --eval-ratio 0.1 --eval-steps 10 --use-vllm --vllm-gpu-memory-utilization 0.3
```

Setting `--epochs` without `--max-steps` sets `max_steps: -1` so Hugging Face
runs by epoch count.

Unknown YAML keys are ignored with a warning (`Ignoring unknown config key …`).
Top-level `preset:` / `scope:` map onto `experiment.preset` / `experiment.scope`.
When `output_dir` (all four blocks) and `processor.runtime_dir` are not set in
the YAML, they follow the resolved working dir (`{working_dir}/models`,
`{working_dir}/runtime`), also with `SEBENI_HOME` plus `-w`.

## `train` flags

| Flag | Writes to |
| --- | --- |
| `--algorithm` | `algorithm` (`sft` / `grpo` / `dpo` / `apo`) |
| `--lang` | `data.languages`, `experiment.scope` |
| `--hitl/--no-hitl` | `distillation.hitl` (initial distill only) |
| `--lr` | active block `learning_rate` |
| `--batch-size`, `--per-device-train-batch-size` | active block `per_device_train_batch_size` |
| `--grad-accum`, `--gradient-accumulation-steps` | active block `gradient_accumulation_steps` |
| `--max-steps` | active block `max_steps` |
| `--epochs`, `--num-train-epochs` | active block `num_train_epochs` |
| `--beta` | active block `beta` (GRPO / DPO / APO) |
| `--warmup-ratio`, `--warmup-steps`, `--weight-decay`, `--seed`, `--save-steps` | active block, same name |
| `--max-prompt-length` | active block `max_prompt_length` (GRPO / DPO / APO) |
| `--optim` | `trainer.optim` (only GRPO has the field) |
| `--lr-scheduler` | active block `lr_scheduler_type` |
| `--bf16/--no-bf16`, `--fp16/--no-fp16`, `--use-cpu/--no-use-cpu` | active block |
| `--eval-steps` | active block `eval_steps`, and sets `eval_strategy: steps` |
| `--num-generations`, `--max-completion-length`, `--temperature` | `trainer:` only (GRPO) |
| `--use-vllm/--no-use-vllm`, `--vllm-gpu-memory-utilization` | `trainer:` only (GRPO) |
| `--grad-checkpoint/--no-grad-checkpoint` | `gradient_checkpointing` on all four blocks |
| `--lora-r`, `--lora-alpha`, `--lora-dropout` | `model:` |
| `--load-in-4bit/--no-load-in-4bit`, `--peft/--no-peft` | `model.load_in_4bit`, `model.use_peft` |
| `--eval-ratio` | `data.eval_ratio` |
| `--eval-rows` | `data.eval_rows` (overrides `--eval-ratio`) |
| `--kveritas/--no-kveritas` | `experiment.kveritas` (`--no-kveritas` also turns sealing off) |
| `--kveritas-seal/--no-kveritas-seal` | `experiment.kveritas_seal` for this run |

"Active block" is `trainer:` for GRPO, `sft:` for SFT, `dpo:` for DPO, `apo:`
for APO. A flag is skipped when the active block has no such field.

The device, `bf16`, and `fp16` follow the active block. When CUDA is missing,
Sebeni falls back to CPU and sets `use_cpu` on all four blocks.

## `model:` (LoRA / quantization)

| Key | Default | CLI |
| --- | --- | --- |
| `model_name` | `HuggingFaceTB/SmolLM2-135M` | (YAML) |
| `ref_model_name` | `null` | (YAML) |
| `flax_model_name` | `null` | (YAML; Flax checkpoint for `framework: jax`) |
| `load_in_4bit` | `true` | `--load-in-4bit` / `--no-load-in-4bit` |
| `load_in_8bit` | `false` | (YAML) |
| `bnb_4bit_quant_type` | `nf4` | (YAML) |
| `bnb_4bit_use_double_quant` | `true` | (YAML) |
| `use_peft` | `true` | `--peft` / `--no-peft` |
| `lora_r` | `16` | `--lora-r` |
| `lora_alpha` | `32` | `--lora-alpha` |
| `lora_dropout` | `0.1` | `--lora-dropout` |
| `lora_target_modules` | `q_proj v_proj k_proj o_proj` | (YAML list) |
| `lora_bias` | `none` | (YAML) |

## `trainer:` (GRPO / TRL `GRPOConfig`)

`framework: torch` (default) uses TRL. `framework: jax` uses Flax/Optax for
GRPO, DPO, or APO when Flax weights exist. If the selected repository has no
Flax weights, Sebeni explains the failed probe and continues with torch/TRL;
set `model.flax_model_name` to pin a separate Flax checkpoint.
Unknown TRL constructor names are dropped at train time so the same YAML
works on TRL 0.x (`max_prompt_length`, `warmup_ratio`) and TRL 1.x
(`max_completion_length`, `warmup_steps`). DPO maps `max_prompt_length`
onto `max_length` when the installed TRL no longer has the prompt field.

| Key | Default | CLI |
| --- | --- | --- |
| `learning_rate` | `5e-6` | `--lr` |
| `per_device_train_batch_size` | `2` | `--batch-size` |
| `gradient_accumulation_steps` | `8` | `--grad-accum` |
| `max_prompt_length` | `1024` | `--max-prompt-length` (ignored on TRL 1.x GRPO; kept for 0.x / DPO mapping) |
| `max_completion_length` | `1024` | `--max-completion-length` |
| `max_steps` | `10` | `--max-steps` |
| `num_train_epochs` | `1.0` | `--epochs` |
| `logging_steps` | `1` | (YAML) |
| `save_steps` | `50` | `--save-steps` |
| `save_total_limit` | `2` | (YAML) |
| `max_grad_norm` | `0.1` | (YAML) |
| `beta` | `0.1` | `--beta` |
| `num_generations` | `4` | `--num-generations` |
| `num_iterations` | `1` | (YAML) |
| `temperature` | `0.9` | `--temperature` |
| `top_p` | `1.0` | (YAML) |
| `top_k` | `50` | (YAML) |
| `warmup_ratio` | `0.0` | `--warmup-ratio` (ignored on TRL 1.x; use `warmup_steps`) |
| `warmup_steps` | `0` | `--warmup-steps` |
| `weight_decay` | `0.0` | `--weight-decay` |
| `lr_scheduler_type` | `cosine` | `--lr-scheduler` |
| `seed` | `42` | `--seed` |
| `optim` | `adamw_torch` | `--optim` |
| `bf16` | `false` | `--bf16` / `--no-bf16` |
| `fp16` | `false` | `--fp16` / `--no-fp16` |
| `gradient_checkpointing` | `false` | `--grad-checkpoint` / `--no-grad-checkpoint` |
| `dataloader_num_workers` | `0` | (YAML) |
| `use_cpu` | `false` | `--use-cpu` / `--no-use-cpu` |
| `framework` | `torch` | (YAML; `torch` or `jax`) |
| `use_vllm` | `false` | `--use-vllm` / `--no-use-vllm` |
| `vllm_gpu_memory_utilization` | `0.3` | `--vllm-gpu-memory-utilization` |
| `vllm_mode` | `colocate` | (YAML) |
| `vllm_device` | `null` | (YAML) |
| `eval_strategy` | `null` | (YAML; set to `steps` by `--eval-steps` or when an eval split exists) |
| `eval_steps` | `null` | `--eval-steps` (auto: `max_steps // 10` when an eval split exists) |
| `per_device_eval_batch_size` | `1` | (YAML; rounded up by GRPO, see below) |
| `report_to` | `trackio` | (YAML; passed through to TRL with `project=<project_name>` so Trainer logs into that Trackio project, not `huggingface`. Run name defaults to `run-YYYYmmdd-HHMMSS`.) |
| `push_to_hub` | `false` | (YAML) |
| `hub_model_id` | `null` | (YAML) |
| `hub_strategy` | `end` | (YAML) |
| `hub_token` | `null` | (YAML) |
| `hub_private_repo` | `true` | (YAML) |

GRPO adjusts settings at train time and prints a notice when it does:

- vLLM is turned off for Gemma-4 models and when CUDA or the `vllm` package is
  missing; generation falls back to Hugging Face.
- When `per_device_train_batch_size × gradient_accumulation_steps` is not
  divisible by `num_generations`, `gradient_accumulation_steps` is raised to
  the smallest value that is.
- When an eval split exists, `per_device_eval_batch_size` is rounded up so
  `per_device_eval_batch_size × WORLD_SIZE` is divisible by
  `num_generations_eval or num_generations` (TRL's default of 8 when
  `num_generations` is unset). A `[GRPO Guard]` line reports the change.

## `sft:` (TRL `SFTConfig`)

Used when `algorithm: sft`. SFT is torch-only.

| Key | Default |
| --- | --- |
| `learning_rate` | `5e-6` |
| `per_device_train_batch_size` | `2` |
| `gradient_accumulation_steps` | `8` |
| `max_length` | `2048` |
| `max_steps` | `10` |
| `num_train_epochs` | `1.0` |
| `logging_steps` / `save_steps` | `1` / `50` |
| `max_grad_norm` | `0.1` |
| `warmup_ratio` / `weight_decay` | `0.0` / `0.0` |
| `lr_scheduler_type` / `seed` | `cosine` / `42` |
| `bf16` / `fp16` / `gradient_checkpointing` / `use_cpu` | `false` |
| `report_to` | `trackio` |
| `eval_strategy` / `eval_steps` / `per_device_eval_batch_size` | `null` / `null` / `1` |
| `push_to_hub` / `hub_model_id` / `hub_token` / `hub_private_repo` | `false` / `null` / `null` / `true` |

SFT trains on prompt / completion pairs: the prompt is the same chat prompt
GRPO, DPO, and eval use, and the loss covers the answer only.

Rows whose prompt + answer is longer than `max_length` (SFT, DPO, APO) are
**dropped, not truncated**, so the JSON target is never cut mid-object. A
warning reports how many were dropped, and training stops with an error when
no row fits. Counts go to `exp/sft_label_report.json` and the safety snapshot.
DabaX targets are long (hundreds to thousands of tokens), so set `max_length`
accordingly.

## `dpo:` / `apo:` (TRL `DPOConfig`)

Same optimizer fields as `sft:` plus:

| Key | `dpo:` default | `apo:` default |
| --- | --- | --- |
| `beta` | `0.1` | `0.1` |
| `loss_type` | `sigmoid` | `apo_zero` |
| `max_prompt_length` | `1024` | `1024` |
| `max_length` | `2048` | `2048` |
| `eval_accumulation_steps` | `1` | `1` |

`algorithm: sft | grpo | dpo | apo` selects the arm.
DPO uses the `dpo:` block (`loss_type: sigmoid`); APO ([Anchored Preference Optimization, arXiv:2408.06266](https://arxiv.org/abs/2408.06266)) uses `apo:`.
APO anchors policy likelihoods relative to the reference model to prevent reward drift and underspecification:
- `loss_type: apo_zero`: Anchors win/loss likelihoods to zero drift ($\mathcal{L} = -\sigma(r_\theta(x, y_w)) + \sigma(r_\theta(x, y_l))$). Increases winning completion likelihoods while decreasing losing completion likelihoods. Recommended when target model capability is below the training set preferences.
- `loss_type: apo_down`: Pushes down both chosen and rejected completions ($\mathcal{L} = \sigma(r_\theta(x, y_w)) - \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$) with stronger downward force on rejected completions. Recommended when the base model is already strong to prevent regression.

## `data:` (multilingual)

| Key | Default | Meaning |
| --- | --- | --- |
| `languages` | `null` | List or comma-separated ISO / group codes for this run |
| `default_lang` | `bam` | Fallback for unlabeled rows |
| `source` | `null` | Path, glob, list of paths, or Hugging Face id |
| `scheme` | `completion` | `completion` (GRPO / SFT), `preference` (DPO/APO), `online_group` |
| `eval_ratio` | `0.1` | Fraction of training rows held out for in-training eval |
| `eval_rows` | `null` | Row count held out for in-training eval; wins over `eval_ratio`; `0` disables the split |
| `encoding` | `utf-8-sig` | File encoding |
| `file_text_key` / `file_lang_key` | `text` / `lang` | JSONL / CSV column names |
| `infer_lang_from_filename` | `true` | `.txt` files: take `lang` from the file name |
| `auto_detect_lang` | `false` | Guess `lang` for unlabeled rows (needs `langdetect`) |
| `max_text_len` | `null` | Truncate longer texts |
| `strip_text` / `skip_empty` / `lowercase_lang` | `true` | Row cleanup |
| `hf_config_name`, `hf_split` (`train`), `hf_text_key`, `hf_lang_key`, `hf_streaming`, `hf_kwargs` | | Hugging Face dataset loading |

`--lang` on `init`, `train`, `distill`, `eval`, and `wordfreq` is repeatable.

`sebeni exp` trains on `data.source` when the `-c` config sets it, otherwise on
the packaged `dataset_300_samples.jsonl` (`--preset` always uses the packaged
file). `sebeni eval` scores `experiment.eval_source` (or `--eval-file`) when
set, otherwise the packaged `test.json`.
See [Experiments](experiments.md).

## `experiment:`

| Key | Default | Meaning |
| --- | --- | --- |
| `max_eval_rows` | `null` | Held-out rows decoded per language by `eval`; `null` or `0` scores every row (negative is an error). Does not size the in-training split (`data.eval_ratio` / `data.eval_rows` do) |
| `eval_source` | `null` | Held-out rows for `eval`: path, glob, list, or Hugging Face id (like `data.source`); relative paths resolve against the config file's directory. `null` = packaged `test.json`. CLI: `eval --eval-file` |
| `eval_max_new_tokens` | `512` | Greedy decoding budget per held-out row (`eval`, `generate`) |
| `kveritas` | `false` | Re-launch `exp` / `train` / `eval` once under `kveritas run` and stream `KV …` lines (also `--kveritas`) |
| `kveritas_seal` | `false` | Seal a report: `exp/report.pdf` (`exp`), `exp/report_train.pdf` (`train`), `exp/report_eval.pdf` (`eval`) |
| `kveritas_disclosure` | `open` | `redacted`, `names`, or `open` |
| `kveritas_local` | `false` | Sign with a local key instead of an attestation server |
| `kveritas_local_key` | `null` | PEM path; default `{working_dir}/keys/private.pem` (`openssl genrsa -out <path> 3072`) |
| `freeze_resources` | `true` | Reuse `data/resources/frozen.json` instead of re-distilling in `train` |
| `dataset` | `raw` | Used only when `data.source` is unset: `raw` reads `beni/data/raw/*.txt`; `packaged` reads `dataset_300_samples.jsonl` (set by `exp` when `data.source` is unset) |
| `scope` | `null` | Set from `--lang` (`MULTI13`, `SINGLE_LANG`, `OUTLIER`, `PARTIAL`) |
| `preset` | `null` | Informational |

## `distillation:`

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Run Distiller before training |
| `backend` | `algorithmic` | `algorithmic`, `gguf`, `google` (alias `gemini`), `openai`, `groq`, `together` |
| `provider` | `null` | Deprecated alias of `backend` |
| `model` | `gemini-2.5-flash` | LLM backend model |
| `tau` | `0.5` | Φ threshold |
| `hitl` | `false` | Review G, D on a TTY |
| `vertex` | `null` | `true` forces Vertex AI / ADC; `null` auto-selects |
| `location` | `global` | Vertex location |
| `project_id` | `null` | GCP project (else env / ADC) |
| `base_url` | `null` | OpenAI-compatible endpoint |
| `gguf_path`, `n_ctx` | `null`, `4096` | Local GGUF backend |
| `max_input_chars` | `8000` | Prompt input cap |
| `batch_size` | `10` | Distiller batch size |
| `auto_update_baselines` | `true` | Write promoted `baseline_vN` |

Details: [Distillation](distillation.md).

## What the CLI does *not* replace

Reward weights, SafetyGovernor gates (for example `safety.require_eval_metrics`,
default `true`: `sebeni push` needs a snapshot from `sebeni eval` with Φ plus
MER or MCS), Distiller `backend` / `model` / `tau`, `experiment.max_eval_rows`,
and Hub tokens stay in YAML (or the environment for API keys).
