# Use cases

Each example is a full path from data → SAMPG → artifacts. Swap model names
and GPU flags for your hardware. Commands assume `sebeni[train,distil]`.
Language codes: [Languages](languages.md). Zero-data-prep run:
[Experiments](experiments.md).

## 1. Batteries-included multilingual (`sebeni exp`)

Edit `configs/exp.yaml` (model, hyperparams, `report_to`) and run on packaged
raw / test splits — no jsonl to prepare.

```bash
sebeni exp -c configs/exp.yaml -w ./runs/exp-001
```

Expect: `{working_dir}/exp/eval.json` with MER, MCS, and UWEC, frozen G/D under
`data/resources/{lang}/`, policy under `models/`, Trackio (or wandb). Details:
[Experiments](experiments.md).

## 2. Bambara GRPO

Post-train an SLM so JSON analyses of Bambara sentences raise Φ.

Data (`data/bam.jsonl`):

```json
{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}
{"text": "N bɛ taa so.", "lang": "bam"}
```

```bash
# No API key is needed for the default algorithmic Distiller.
sebeni init --lang bam -w ./runs/bam-grpo
# edit config: data.source, model.model_name, trainer.max_steps
sebeni train -c ./runs/bam-grpo/config.yaml \
  --lr 5e-6 --batch-size 2 --num-generations 4 --max-steps 50
sebeni eval -c ./runs/bam-grpo/config.yaml
sebeni wordfreq -c ./runs/bam-grpo/config.yaml
```

YAML fragment:

```yaml
algorithm: grpo
data:
  default_lang: bam
  languages: [bam]
  source: ./data/bam.jsonl
  scheme: completion
model:
  model_name: HuggingFaceTB/SmolLM2-135M
  use_peft: true
  lora_r: 16
trainer:
  beta: 0.1
  num_generations: 4
  max_steps: 50
distillation:
  backend: algorithmic
  tau: 0.5
```

Expect: `{working_dir}/data/baselines/bam/baseline_vN.{gram,dict}`,
policy under `models/`, `exp/eval.json` with Φ, `exp/wordfreq/bam/`.

## 3. Multilingual Manding (one policy, many G/D)

Train **one** θ on mixed Bambara + Maninka + Dogon. Each row keeps `lang`.
Φ and Distiller run **per group**.

```json
{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}
{"text": "i ni ce", "lang": "mku"}
{"text": "…", "lang": "dtm"}
```

```bash
sebeni init --lang bam --lang mku --lang dtm -w ./runs/manding
sebeni train -c ./runs/manding/config.yaml --lr 1e-5 --lora-r 32
sebeni eval -c ./runs/manding/config.yaml
sebeni generate -c ./runs/manding/config.yaml \
  --prompt "Aw ka kɛnɛ wa?" --lang bam
```

See `configs/grpo_multilang.yaml`. `R_lang` is 0 if the JSON `lang` does not
match **that row** (Maninka must be `mku`, not `mlq`, unless you rely on
alias resolution — prefer `mku`).

## 4. Language with no packaged grammar (scratch bootstrap)

For a code with no `beni/data/baselines/{lang}/`, Distiller writes Daba
stubs and uses **bootstrap** prompts (full G, D — not `[ADD]`/`[REPLACE]`).
First promote is “are the files parseable?”; later `baseline_vN` requires
Φ′ > Φ.

```bash
sebeni init --lang zzz -w ./runs/scratch-zzz
# data.source: a small jsonl with lang: zzz
sebeni train -c ./runs/scratch-zzz/config.yaml --hitl
# TTY: review G, D
# CI / non-TTY: HITL is skipped automatically
```

## 5. DPO / APO from ranked groups

When you already have group completions + scores (or `chosen` / `rejected`),
set `algorithm: dpo` or `apo`. SAMPG (Φ / Distiller) is unchanged.

```yaml
algorithm: dpo           # or apo
data:
  scheme: preference
dpo:
  beta: 0.1
  loss_type: sigmoid     # apo: apo_zero | apo_down
```

```bash
sebeni train -c configs/dpo_bam.yaml -w ./runs/bam-dpo
sebeni train -c configs/apo_bam.yaml -w ./runs/bam-apo
```

Records may be `{text, lang, chosen, rejected}` or
`{text, lang, completions, scores}`.

## 6. Eval / wordfreq without training

Point `data.source` at a held-out split for eval. Set `wordfreq.raw_inputs` for
the raw-text frequency pipeline and run without a
policy step:

```bash
sebeni eval -c config.yaml --lang bam --lang mku
sebeni wordfreq -c config.yaml
```

`exp/eval.json` contains `phi`, `by_language`, `n_sentences`. Compare MER/MCS
when you have IGT references (`MorphologyScorer.mer` / `.mcs`).

## 7. Publish a Hub model

After a successful `train` (card + `safety_snapshot.json` required):

```bash
huggingface-cli login    # write token for mlsftwrs
sebeni push -c config.yaml --repo-id mlsftwrs/sebeni-bam-grpo
```

Org transfer and tokens: [Hub](hub.md).

## 8. CPU smoke test (CI / laptop)

```yaml
trainer:
  use_cpu: true
  max_steps: 1
  per_device_train_batch_size: 1
  num_generations: 2
model:
  load_in_4bit: false
  use_peft: true
distillation:
  enabled: false     # skip Distiller if you only test the CLI
```

```bash
sebeni train -c config.yaml --use-cpu --max-steps 1 --no-load-in-4bit
```

## 9. All 13 national languages

One policy, thirteen `{G, D}` checkpoints:

```bash
sebeni init \
  --lang bam --lang bmq --lang boz --lang dtm --lang ful \
  --lang mey --lang kao --lang myk --lang mku --lang spp \
  --lang ses --lang snk --lang taq \
  -w ./runs/mali-13
```

JSONL (one language identity **per row**):

```json
{"text": "…", "lang": "bam"}
{"text": "…", "lang": "mku"}
{"text": "…", "lang": "taq"}
```

Start from `configs/grpo_multilang.yaml` and set `data.languages` to the
thirteen group codes. Distiller + Φ still run **per group**; `R_lang` still
scores **that row**. Larger `num_generations` needs more GPU memory — treat
that as a hardware knob, not a result.

## 10. Distill G, D then align later

Useful when Distiller is rate-limited or you want review before GRPO:

```bash
sebeni distill -c config.yaml --hitl
# inspect {working_dir}/data/baselines/bam/baseline_vN.*
sebeni train -c config.yaml --no-hitl --max-steps 100
```

Standalone `sebeni distill` vs in-loop Distiller: [Distillation](distillation.md).

## 11. Programmatic SAMPG

```python
from beni.core.srl.config import MasterConfig
from beni.core.srl.unified import SRLTrainer
from beni.core.srl.algorithm1 import maybe_distill_languages
from beni.core.safety import SafetyGovernor

cfg = MasterConfig.from_yaml("config.yaml")
gov = SafetyGovernor(cfg.safety.to_spec(tau=cfg.distillation.tau, kl_beta=cfg.trainer.beta))
# Optional: Distiller-only pass on mixed rows
# maybe_distill_languages(pairs, distiller_for, gov, tau=0.5)
trainer = SRLTrainer(cfg)
trainer.train([
    {"text": "Aw ka kɛnɛ wa?", "lang": "bam"},
    {"text": "i ni ce", "lang": "mku"},
])
```
