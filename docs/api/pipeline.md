# `beni.core.pipeline`

The canonical pipeline. `sebeni distill`, `train`, `eval`, and `exp` all call
these functions: distill once, train one arm, evaluate held-out data.

```python
from beni.core.srl.config import MasterConfig
from beni.core.pipeline import (
    load_policy, run_distill, run_arm, run_eval, run_experiment,
)

mc = MasterConfig.from_yaml("config.yaml")
run_distill(mc)                 # freeze G, D → data/resources/frozen.json
trainer = run_arm(mc)           # one arm; distills first if not frozen
policy = load_policy(mc)        # trained weights from the arm's output dir
report = run_eval(mc, generate=lambda text, lang: policy.analyze(text),
                  policy=getattr(policy, "plugin", policy),
                  weights=policy.loaded_weights)
# or: run_experiment(mc)        # distill + arm + eval
```

`run_eval(mc, records=None, generate=None, policy=None, weights=None,
record=True)` reads held-out rows from `experiment.eval_source` when set,
otherwise packaged `test.json` (`load_eval_records`). `weights` records which
trained weights were scored: `eval.json` stores it, and the safety snapshot is
written next to those weights (with their `weights_sha256`); without weights it
goes to `{working_dir}/exp/`. `record=False` (in-training probes) only returns
the report: no `eval.json`, snapshot, manifest, or K-Veritas lines. Without
`generate` / `policy` nothing is decoded, so `mer`, `mcs`, and `uwec` come back
`null` and Φ falls back to the reference parse.

`load_policy(mc, weights=None, require=False)` builds the arm's trainer and
loads trained weights: `weights` (a directory or Hub repo id) or, by default,
`resolve_policy_weights(mc)`, which returns the arm's output dir
(`default_weights_dir(mc)`, usually `{working_dir}/models`) when it holds
`adapter_config.json` or `config.json`. A LoRA adapter is attached to a fresh
base model; a full checkpoint is loaded directly. Without trained weights it
loads the base model (`trainer.loaded_weights` is `None`), or raises
`FileNotFoundError` when `require=True`.

`run_experiment` trains on `data.source` when the config sets it, otherwise on
the packaged `dataset_300_samples.jsonl`, and prints
`[exp] train data: <source> (<n> rows; …)`. It then evaluates the arm's output
dir.

::: beni.core.pipeline
    options:
      members:
        - run_distill
        - run_arm
        - run_eval
        - run_experiment
        - load_policy
        - resolve_policy_weights
        - default_weights_dir
        - load_train_records
        - load_eval_records
        - resources_frozen
        - frozen_marker
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: true
