# `beni.core.compute`

Phi, MER, MCS, UWEC, and the reward manager. Held-out MER / MCS / UWEC are
NumPy or JAX reductions. MER and UWEC are costs (lower is better); MCS is a
stage-match fraction (higher is better) in both `eval.json` and the training
rewards. Eval pools counts with `mer_edit_counts` and `mcs_match_counts`.
`mcs_mismatch` (\(1-\mathrm{MCS}\)) is an optional helper the pipeline does not
use.

```python
from beni.core.compute.helpers import mer_micro, mer_edit_counts, mcs_match_counts
from beni.core.compute.metrics import MorphologyScorer
from beni.core.compute.rewards import RewardManager
from beni.core.compute.uwec import uwec

scorer = MorphologyScorer()
phi = scorer.phi(sentence)
mer = scorer.mer(["a", "b"], ["a", "c"])
_, mean = uwec([1, -1], [0.4, 0.2], [0.5, 0.5], beta=0.1, eps=1e-8)

rm = RewardManager()
fmt = rm.reward_format(['{"text": "aw", "lang": "bam", "tokens": []}'])
lang = rm.reward_lang(
    ['{"text": "aw", "lang": "bam", "tokens": []}'],
    language=["bam"],
)
```

::: beni.core.compute.metrics.MorphologyScorer
    options:
      members:
        - phi
        - phi_corpus
        - mer
        - mer_sentence
        - mcs
        - mcs_corpus
        - evaluate
        - uncertainty_weighted_cost_error
        - uncertainty_weighted_cost_error_sentence
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false

::: beni.core.compute.uwec.uwec
    options:
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false

::: beni.core.compute.helpers.mer_micro
    options:
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false

::: beni.core.compute.helpers.mer_edit_counts
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.compute.helpers.mcs_match_counts
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.compute.helpers.stage_to_phi
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.compute.helpers.mcs_mismatch
    options:
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false

`RewardManager.compute_rewards` (used by the JAX path) returns the `format`,
`morph`, `rule`, `lang`, and `total` score lists for a batch.

::: beni.core.compute.rewards.RewardManager
    options:
      members:
        - reward_format
        - reward_morph
        - reward_rule
        - reward_lang
        - get_reward_functions
        - register_custom_reward
        - compute_rewards
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false
