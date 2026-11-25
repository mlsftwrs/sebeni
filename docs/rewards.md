# Rewards and metrics

Shared \(\pi_{rf}\) for every algorithm:

\[
R_{\mathrm{total}} = \omega_m R_{\mathrm{morph}} + \omega_r R_{\mathrm{rule}} + \omega_f R_{\mathrm{format}} + \omega_l R_{\mathrm{lang}}
\]

Dataclass and `sebeni init` defaults (YAML `reward:`): morph **0.4**, rule
**0.4**, format **0.1**, lang **0.1**. The packaged preset `beni/data/presets/multi13.yaml`
(what `sebeni exp` loads with no `-c`) overrides them to morph **0.3**, rule **0.3**,
format **0.2**, lang **0.2**. Both sum to **1.0**. GRPO stays
**outcome-level** on the completion; token stages are process signal. Daba
itself is the verifiable reward — no separate reward model.

## Metrics

| Metric | Direction | Pooling |
| --- | --- | --- |
| Φ | higher is better | top-level `phi` is a **macro** mean over languages |
| MER | cost, lower is better | micro over morphemes |
| MCS | match fraction, **higher is better** | micro over tokens |
| UWEC | cost, lower is better | micro over tokens |

**Phi** — mean `stage_to_phi` over tokens:

| Stage | Contribution |
| --- | --- |
| Analyzed (`0 .. max`, non-EMPR) | 1.0 |
| EMPR (grammar borrowing stage: 6 in `bam`, 4 in the other 12 baselines; or `gloss == "EMPR"`) | `empr` (default 0.5) |
| Above grammar EMPR stage | `eps` (≈ 0) |
| −1 (unknown) | 0.0 |
| Non-numeric label | 1.0 |

During **training** (`R_morph` / `R_rule`), a completion token with no `"stage"` key is read as −1. During **held-out evaluation** (`sebeni eval` / `run_eval`), any LLM-emitted `"stage"` is discarded and DabaX recomputes each predicted token's stage from its surface. Distiller runs when a
language's train-split Φ < τ (0.5).

**MER** — morpheme edit rate \((S_m+D_m+I_m)/N_m\) from a Levenshtein
traceback. \(N_m\) is the reference morpheme count. The reported value for a
language or MULTI13 is one micro-average over every morpheme in that scope.
Eval compares the **first** DabaX sentence of the reference parse with the
first sentence of the hypothesis. A row whose reference has no morphemes is
skipped (not decoded, kept out of the pools) and counted in `n_skipped`, so
MER never becomes `inf`.

**MCS** — morphosyntactic consistency score:

\[
\mathrm{MCS} = \frac{1}{\max(|T_{pred}|, |T_{ref}|)} \sum_{i=1}^{\min(|T_{pred}|, |T_{ref}|)} \mathbb{I}\big(\mathrm{Stage}_{pred}(w_i) = \mathrm{Stage}_{ref}(w_i)\big)
\]

Tokens are paired **by index** (no alignment); unpaired tokens are not
matches. `eval.json` reports this match fraction, and training rewards use the
same quantity. `helpers.mcs_mismatch` (\(1-\mathrm{MCS}\)) is an optional
helper; the pipeline does not use it. In training rewards (`R_morph`), omitting
`"stage"` yields −1 on every token; in held-out evaluation (`run_eval`),
DabaX computes the token stages regardless of what `"stage"` the LLM emitted.

**UWEC** — post-training cost

\[
\mathrm{UWEC} = \frac{1}{|W|}\sum_{i} \Big( \mathbb{I}(\mathrm{Stage}_{pred}(w_i)\neq-1) + \beta\left|\log\frac{\pi_\theta(w_i)+\varepsilon}{\pi_{\mathrm{ref}}(w_i)+\varepsilon}\right| \Big)
\]

token-mean over the scope. It is not a training gradient coefficient. Torch or
Flax may produce the token probabilities; the reduction is NumPy or JAX.

!!! note "UWEC null / NaN handling"
    - LM token count ≠ morphological token count → every morphological token
      gets the mean sequence probability.
    - Non-finite probabilities are replaced by 1. All-non-positive inputs are
      treated as log-probabilities and exponentiated.
    - If the mean is still non-finite, eval retries with uniform probabilities,
      so UWEC becomes the pure stage indicator for that sentence.
    - With no `ref_model` and no PEFT adapter to disable, \(\pi_{ref}=\pi_\theta\)
      and the log term is 0.
    - `uwec` is `null` only when no sentence was scored. Older published
      eval files with `uwec: null` predate the uniform fallback.

The training distrust term \(U\) used to scale gradients stays separate
(SafetyGovernor, see [Safety](safety.md)).

## `eval.json`

`{working_dir}/exp/eval.json`:

| Level | Fields |
| --- | --- |
| top | `model`, `algorithm`, `scope`, `languages`, `phi`, `mer`, `mcs`, `uwec`, `n_morphemes`, `n_tokens`, `n_scored`, `n_skipped`, `beta`, `eps`, `reference_model`, `weights`, `eval_source`, `max_eval_rows`, `tau`, `by_language` |
| `by_language.{lang}` | `phi`, `mer`, `mcs`, `uwec`, `n_sentences`, `n_scored`, `n_skipped`, `n_morphemes`, `n_tokens`, `checkpoint_id`, `language`, `scope` |

- `n_sentences` counts parsed **reference** sentences in the sample; `n_scored`
  counts decoded rows that were scored; `n_skipped` counts rows whose reference
  has no morphemes.
- Per-language `phi` is computed on the decoded hypotheses' stages: for JSON
  completions, LLM-emitted `"stage"` values are discarded and `DabaX` computes
  each token's stage on the predicted surfaces (keeping JSON `analyses` for
  MER). With nothing decoded it falls back to the reference Φ over the
  evaluated rows.
- `checkpoint_id` names the **G/D** checkpoint, not the policy:
  `baseline_vN` (promoted), `baseline` (workdir copy of the packaged files,
  never promoted), `packaged`, or `none`.
- A completion that is not a JSON object with `tokens` is DabaX-parsed as raw
  text, and those DabaX stages are scored. That can inflate MCS.
- `sebeni eval` scores the **trained** weights: `--weights PATH|REPO`
  (default: the arm's output dir, `{working_dir}/models`). A LoRA adapter is
  attached to a fresh base model, a full checkpoint is loaded directly, and a
  Hub repo id resolves through the local Hugging Face cache. Without trained
  weights, eval warns and scores the base model (`weights: null`).
- Decoding is **greedy** with the **same prompt as training** (system prompt +
  chat template; ChatML fallback when the tokenizer has none), up to
  `experiment.eval_max_new_tokens` (default 512) new tokens, stopping at end of
  turn. The training prompt's languages are saved next to the weights in
  `sebeni_policy.json` and reused, so `eval --lang` only filters rows.
- Rows come from `experiment.eval_source` (or `eval --eval-file`), else the
  packaged `test.json`. `experiment.max_eval_rows` caps the rows decoded per
  language; the default `null` (or `0`) scores every held-out row.

## Rewards

| Term | Default weight | Computation |
| --- | --- | --- |
| \(R_{morph}\) | 0.4 | \(\omega_m \cdot \frac{1}{3}\big(\mathrm{MCS} + \mathrm{clip}(1-\mathrm{MER}, 0, 1) + \text{lemma overlap}\big)\) against the DabaX reference \(y^*\). 0 when either side has no tokens. |
| \(R_{rule}\) | 0.4 | \(\omega_r \cdot 0.5\,\big(\text{consistency} + \text{lemma overlap with } y^*\big)\). Consistency scores the **claimed** analysis against DabaX's parse of the prompt sentence (or the reference column): a token at the same position earns its Φ weight (`stage_to_phi`) only if surface, stage, and first analysis (form + POS) match; the sum is divided by \(\max(\lvert T_{pred}\rvert, \lvert T_{ref}\rvert)\). The completion's `"text"` is never re-parsed, so echoing the input earns nothing. 0 when either side has no tokens. |
| \(R_{format}\) | 0.1 | \(\omega_f\) if the completion yields a JSON object with a `tokens` list, else 0 |
| \(R_{lang}\) | 0.1 | \(\omega_l\) if the JSON `lang` resolves to **that row**'s group code, else 0. A completion declaring several languages (mixed) scores 0. |

JSON is extracted leniently: direct parse, then fenced ```` ```json ```` blocks,
then the first balanced `{…}` object.

`RewardManager.compute_rewards` (used by the JAX path) returns the
`format`, `morph`, `rule`, `lang`, and `total` score lists for a batch.

There are no positional / valence / lexical (ρ, ν, λ) rule terms, no explicit
penalty for claiming a stage Daba rejects (that only lowers MCS inside
\(R_{morph}\)), and no length or insertion penalty.

```json
{"text": "aw ka ne labato.", "lang": "bam", "tokens": [{"surface": "aw", "stage": 1, "analyses": []}]}
```

Invalid JSON, missing `tokens`, or a mismatched `lang` scores 0 for the
corresponding term. For GRPO, the format gate blocks a policy update only when
every completion since the last optimizer step has \(R_{format} = 0\); SFT,
DPO, and APO refuse the step when any training target is invalid — see
[Safety](safety.md).

## YAML

```yaml
reward:
  format_weight: 0.1
  morph_weight: 0.4
  rule_weight: 0.4
  lang_weight: 0.1
  enable_format_reward: true
  enable_morph_reward: true
  enable_rule_reward: true
  enable_lang_reward: true
```

```python
from beni.core.compute.rewards import RewardManager

rm = RewardManager()
fmt = rm.reward_format(['{"text": "aw", "lang": "bam", "tokens": []}'])
lang = rm.reward_lang(
    ['{"text": "aw", "lang": "bam", "tokens": []}'],
    language=["bam"],
)
```
