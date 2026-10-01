# Model and trainer hyperparameters

Every field on `ModelConfig`, `GRPOTrainerConfig`, `DPOTrainerConfig`, and
`APOTrainerConfig` is a YAML key under `model:`, `trainer:`, `dpo:`, or `apo:`.
`sebeni train` overlays the same knobs from the CLI (non-None flags win).

Docs site: [seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).

```bash
sebeni train -c config.yaml \
  --lr 1e-5 --batch-size 4 --grad-accum 4 --max-steps 100 \
  --beta 0.04 --num-generations 8 --temperature 0.8 \
  --lora-r 32 --lora-alpha 64 --bf16 --grad-checkpoint
```

Setting `--epochs` without `--max-steps` sets `max_steps: -1` so Hugging Face
runs by epoch count.

## `model:` (LoRA / quantization)

| Key | Default | CLI |
| --- | --- | --- |
| `model_name` | `HuggingFaceTB/SmolLM2-135M` | (YAML) |
| `ref_model_name` | `null` | (YAML) |
| `load_in_4bit` | `true` | `--load-in-4bit` / `--no-load-in-4bit` |
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
| `gradient_checkpointing` | `false` | `--grad-checkpoint` |
| `dataloader_num_workers` | `0` | (YAML) |
| `use_cpu` | `false` | `--use-cpu` / `--no-use-cpu` |
| `report_to` | `trackio` | (YAML; passed through to TRL with `project=<project_name>` so Trainer logs into that Trackio project, not `huggingface`.) |
| `push_to_hub` | `false` | (YAML) |
| `hub_model_id` | `null` | (YAML) |

`algorithm: sft | grpo | dpo | apo` selects the arm. SFT is torch-only.
DPO uses the `dpo:` block (`loss_type: sigmoid`); APO ([Anchored Preference Optimization, arXiv:2408.06266](https://arxiv.org/abs/2408.06266)) uses `apo:`.
APO anchors policy likelihoods relative to the reference model to prevent reward drift and underspecification:
- `loss_type: apo_zero`: Anchors win/loss likelihoods to zero drift ($\mathcal{L} = -\sigma(r_\theta(x, y_w)) + \sigma(r_\theta(x, y_l))$). Increases winning completion likelihoods while decreasing losing completion likelihoods. Recommended when target model capability is below the training set preferences.
- `loss_type: apo_down`: Pushes down both chosen and rejected completions ($\mathcal{L} = \sigma(r_\theta(x, y_w)) - \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$) with stronger downward force on rejected completions. Recommended when the base model is already strong to prevent regression.

Shared optimizer fields (`learning_rate`, `beta`, `max_steps`, `warmup_ratio`, `weight_decay`, `lr_scheduler_type`, `seed`, `bf16`, `fp16`) overlay onto the active algorithm when you pass CLI flags.

## `data:` (multilingual)

| Key | Meaning |
| --- | --- |
| `languages` | List or comma-separated ISO / group codes for this run |
| `default_lang` | Fallback for unlabeled rows |
| `source` | Path, glob, list of paths, or Hugging Face id |
| `scheme` | `completion` (GRPO) or `preference` (DPO/APO) |

`--lang` on `init`, `train`, `distill`, `eval`, and `wordfreq` is repeatable.

`sebeni exp` ignores `data.source` and always uses packaged raw / test splits.
See [Experiments](experiments.md).

## What the CLI does *not* replace

Reward weights, SafetyGovernor gates, Distiller `backend` / `model` / `tau`,
and Hub tokens stay in YAML (or the environment for API keys).
