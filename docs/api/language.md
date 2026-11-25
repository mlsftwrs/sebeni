# `beni.core.language`

```python
from beni.core.language import (
    Language, parse_lang_codes, resolve_scope, experiment_group_code, MALI13,
)

Language.from_code("mlq").group_code   # 'mku'
Language.group_codes("bam,mku,dtm")    # ['bam', 'mku', 'dtm']
parse_lang_codes(["bam", "mku,dtm"])
resolve_scope("bam,bbo").label         # 'SINGLE_LANG' (bbo dropped)
experiment_group_code("mlq")           # 'kao' (experiment jsonl only)
len(MALI13)                            # 13
```

::: beni.core.language.Language
    options:
      members:
        - from_code
        - group_codes
      show_root_heading: true
      heading_level: 2
      merge_init_into_class: true
      show_if_no_docstring: false

::: beni.core.language.parse_lang_codes
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.language.resolve_scope
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.language.experiment_group_code
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.language.MALI13
    options:
      show_root_heading: true
      heading_level: 2
