# `beni.core.compute`

Phi, MER, MCS, UWEC, and the reward manager. Held-out MER / MCS / UWEC are
NumPy or JAX reductions. Training rewards still read MCS as a match fraction.

```python
from beni.core.compute.helpers import mer_micro, mcs_mismatch
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

::: beni.core.compute.helpers.mcs_mismatch
    options:
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false

::: beni.core.compute.rewards.RewardManager
    options:
      members:
        - reward_format
        - reward_morph
        - reward_rule
        - reward_lang
        - get_reward_functions
        - compute_rewards
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false
