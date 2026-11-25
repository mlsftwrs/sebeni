# API reference

Hand-written examples plus **selected** mkdocstrings members. Dumping every
nested dataclass / Typer object onto one page produced unreadable
("broken-down") API HTML; each module now lists the public surface only.

Browse from [mlsftwrs.github.io/sebeni/api/](https://mlsftwrs.github.io/sebeni/api/).

| Page | Symbols |
| --- | --- |
| [Pipeline](pipeline.md) | `run_distill`, `run_arm`, `run_eval`, `run_experiment` |
| [SRL](srl.md) | `MasterConfig`, `SRLTrainer`, `register_algorithm`, arm classes, `JaxPolicyPlugin`, SAMPG helpers |
| [Safety](safety.md) | `SafetySpec`, `SafetyGovernor`, `PromoteDecision`, `SafetySnapshot` |
| [Compute](compute.md) | `MorphologyScorer`, `uwec`, `mer_micro`, `mer_edit_counts`, `mcs_match_counts`, `stage_to_phi`, `RewardManager` |
| [CLI](cli.md) | `sebeni` commands + `default_config_yaml` |
| [Language](language.md) | `Language`, `parse_lang_codes`, `resolve_scope`, `experiment_group_code`, `MALI13` |
| [Distiller / DabaX](morphotactic.md) | `Distiller`, `DabaX` |

Import config **without** torch:

```python
from beni.core.srl.config import MasterConfig
from beni.core.srl.unified import SRLTrainer

cfg = MasterConfig.from_yaml("config.yaml")
SRLTrainer(cfg).train([{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}])
```
