# `beni.core.srl`

SAMPG driver and policy-update plugins. Import config **without** torch:

```python
from beni.core.srl.config import MasterConfig
from beni.core.srl.unified import SRLTrainer, register_algorithm

cfg = MasterConfig.from_yaml("config.yaml")
trainer = SRLTrainer(cfg)          # plugin = SFT / GRPO / DPO / APO
trainer.train([{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}])
# trainer.load_models / save_model / generate / push_to_hub → plugin
```

Register a custom policy-update plugin:

```python
register_algorithm("my_gc", MyPlugin)  # YAML algorithm: my_gc
```

::: beni.core.srl.config.MasterConfig
    options:
      members:
        - from_yaml
        - from_dict
        - languages
        - apply_cli_overrides
        - apply_workdir
      show_root_heading: true
      heading_level: 2
      merge_init_into_class: false
      show_if_no_docstring: false

::: beni.core.srl.unified.SRLTrainer
    options:
      members:
        - train
      show_root_heading: true
      heading_level: 2
      inherited_members: false
      show_if_no_docstring: false

::: beni.core.srl.unified.register_algorithm
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.srl.algorithm1.maybe_distill_languages
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.srl.algorithm1.maybe_distill_batch
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.srl.algorithm1.group_texts_by_language
    options:
      show_root_heading: true
      heading_level: 2
