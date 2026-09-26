# Rewards and metrics

Shared \(\pi_{rf}\) for every algorithm:

\[
R_{\mathrm{total}} = \omega_m R_{\mathrm{morph}} + \omega_r R_{\mathrm{rule}} + \omega_f R_{\mathrm{format}} + \omega_l R_{\mathrm{lang}}
\]

Sebeni defaults (YAML `reward:`): morph **0.4**, rule **0.4**, format **0.1**,
lang **0.1** — they sum to **1.0**. GRPO stays **outcome-level** on the
completion; token stages are process signal. Daba itself is the verifiable
reward — no separate reward model.

## Metrics

**Phi** — mean `stage_to_phi` on tokens. Analyzed stages (0–5) → 1.0;
borrowing EMPR / stage 6 → `empr` (default 0.5); unknown → 0. Trigger Distiller
if corpus/batch Φ < τ (0.5).

**MER** — held-out morpheme edit rate \((S_m+D_m+I_m)/N_m\). \(N_m\) is the
reference morpheme count. The reported value for a language or MULTI13 is one
micro-average over every morpheme in that scope. Lower is better.

**MCS (reward)** — fraction of tokens whose predicted `stage` equals Daba's
stage. Training rewards still use this match fraction.

**MCS (report)** — the mismatch cost \(1 -\) that fraction, so MER, MCS, and
UWEC are all quantities to minimize. Unpaired tokens count as mismatches.

**UWEC** — post-training cost
\(I(\mathrm{Stage}_{pred}\neq-1)+\beta\left|\log\frac{\pi^\mathcal{A}_\theta+\varepsilon}{\pi_{\mathrm{ref}}+\varepsilon}\right|\),
token-mean over the scope. It is not a training gradient coefficient. Torch or
Flax may produce the token probabilities; the cost reduction is NumPy or JAX.

The training distrust term \(U\) used to scale gradients stays separate
(`jax_uncertainty_scale` / SafetyGovernor). Evaluation logs MER, MCS, and UWEC
on `{working_dir}/exp/eval.json` as `model`, `algorithm`, `scope`.

## Rewards

| Term | Weight | Meaning |
| --- | --- | --- |
| \(R_{morph}\) | 0.4 | Annotation quality vs DabaX \(y^*\): MCS, clipped MER quality, and lemma overlap |
| \(R_{rule}\) | 0.4 | Text Φ under active **G**, **D**, plus lexical/POS agreement with \(y^*\) |
| \(R_{format}\) | 0.1 | Valid **JSON** object with a `tokens` list |
| \(R_{lang}\) | 0.1 | JSON `lang` matches **that row**'s group |

### Rule-encoded extras

\(R_{rule}\) checks finite-state transitions in **G**. A suffix without a valid
preceding stem is a penalty.

| Parameter | Meaning |
| --- | --- |
| \(\rho\) positional | `:n: [ :v: :n: ]` (verb nominalized as noun) → \(\rho=+1.0\). Nominalizing suffix on a non-verbal stem with no rule in \(G'\) → \(\rho=-1.0\). |
| \(\lambda\) lexical | `\lx` + `\ps` in **D** → 1.0. `\va` / `\fa` variant correctly tagged → 0.8. Phonologically plausible but missing lemma → −0.5. |
| \(\nu\) valence | If a verb is `\vl` transitive in **D**, the window must contain an object marker; mismatch lowers that completion's GRPO advantage \(A_i\). |

### Format / self-awareness

Predicted `stage` matching Daba is rewarded; claiming `stage=1` when Daba says
−1 is penalized. Sebeni encodes this as JSON schema validity plus MCS when a
reference parse exists.

```json
{"text": "aw ka ne labato.", "lang": "bam", "tokens": [{"surface": "aw", "stage": 1, "analyses": []}]}
```

Invalid JSON, missing `tokens`, or a `lang` that is not this row's group
scores 0 for the corresponding term (and can block the policy update — see
[Safety](safety.md)).

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
