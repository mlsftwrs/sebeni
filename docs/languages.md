# Languages

Mali's 2023 constitution lists **13 national languages**. Sebeni scores Φ and
Distiller checkpoints **per group code**. Completions must declare that row's
group in JSON `lang`.

| Language | ISO you may pass | Sebeni group | Notes |
| --- | --- | --- | --- |
| Bambara (Bamanankan) | bam, bm | **bam** | Packaged baseline + largest raw split |
| Bomu | bmq | **bmq** | Packaged baseline |
| Bozo | boz | **boz** | Family grouped |
| Dogon | dtm | **dtm** | Toro So as emblem |
| Fula / Fulani | ful | **ful** | Packaged baseline |
| Hassaniya | mey, hsy | **mey** | Toolkit uses **mey** |
| Kassonke (Xaasongaxango) | kao | **kao** | Packaged baseline |
| Mamara | myk | **myk** | Packaged baseline |
| Maninka | mku, mlq | **mku** | Packaged Daba files may live under `baselines/mlq/` |
| Seynara / Senoufo | spp, seq | **spp** | Senufo group |
| Songhay | ses | **ses** | Packaged baseline |
| Soninke | snk | **snk** | Packaged baseline |
| Tamasheq | taq | **taq** | Packaged baseline |

Aliases such as `bm` → `bam` and `mlq` → `mku` are resolved by
`Language.from_code`. Pass ISO or group to `--lang`.

```bash
sebeni init --lang multi13 -w ./runs/mali-001
sebeni init --lang bam -w ./runs/bam
sebeni exp --preset single --lang bbo --algorithm grpo -w ./runs/bbo
```

`bbo` is an outlier kept in the experiment file and excluded from MULTI13.
Inside `dataset_300_samples.jsonl` only, `mlq` is read as Kassonke `kao`,
`hsy` as `mey`, and `seq` as `spp`. `Language.from_code("mlq")` is still Maninka `mku`.

```yaml
data:
  default_lang: bam          # unlabeled rows only
  languages: [bam, mku, dtm] # filter + prompt scope
```

```python
from beni.core.language import Language, parse_lang_codes

Language.from_code("mlq").group_code   # 'mku'
Language.group_codes("bam,mku,dtm")    # ['bam', 'mku', 'dtm']
parse_lang_codes(["bam", "mku,dtm"])
```

Packaged train texts live under `beni/data/raw/{group}.txt`. A code with no
baseline (for example `bbo` in the packaged raw split) uses Distiller
**scratch bootstrap**. Adding a new language: [use case 4](use-cases.md#4-language-with-no-packaged-grammar-scratch-bootstrap).

Thirteen-language GRPO is [use case 9](use-cases.md#9-all-13-national-languages).
