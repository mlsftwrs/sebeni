# API reference

Hand-written examples plus **selected** mkdocstrings members. Dumping every
nested dataclass / Typer object onto one page produced unreadable
("broken-down") API HTML; each module now lists the public surface only.

Browse from [https://seben.robotsmali.org/docs/api/](https://seben.robotsmali.org/docs/api/).

| Page | Symbols |
| --- | --- |
| [SRL](srl.md) | `MasterConfig`, `SRLTrainer`, `register_algorithm`, SAMPG helpers |
| [Safety](safety.md) | `SafetySpec`, `SafetyGovernor`, `PromoteDecision`, `SafetySnapshot` |
| [Compute](compute.md) | `MorphologyScorer`, `uwec`, `mer_micro`, `mcs_mismatch`, `RewardManager` |
| [CLI](cli.md) | `sebeni` commands + `default_config_yaml` |
| [Language](language.md) | `Language`, `parse_lang_codes` |
| [Distiller / DabaX](morphotactic.md) | `Distiller`, `DabaX` |

Import config **without** torch:

```python
from beni.core.srl.config import MasterConfig
from beni.core.srl.unified import SRLTrainer

cfg = MasterConfig.from_yaml("config.yaml")
SRLTrainer(cfg).train([{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}])
```
