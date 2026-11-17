# Morphotactic: Distiller and DabaX

Distiller in SAMPG proposes G, D when Φ < τ. DabaX wraps CLI `daba.mparser`
(no wxPython). `backend=` selects `algorithmic` (default, no key), `google` /
`gemini`, `openai`, `groq`, `together`, or `gguf`; `provider=` is a deprecated
alias. The pipeline drives one language through
[`distill_language`](srl.md#beni.core.srl.algorithm1.distill_language), and eval
resolves G, D with `resolve_baseline_files`.

```python
from beni.core.morphotactic.distil.distillation import Distiller
from beni.core.morphotactic.dabax import DabaX

d = Distiller(lang_code="bam", backend="algorithmic", working_dir="./runs/bam")
d.handle_baselines()
phi = d.phi_on_texts(["Aw ka kɛnɛ wa?"])

dx = DabaX("bam", gram=d.gram_path, ldict=d.dict_path, process=True)
sentences = dx.loader("Aw ka kɛnɛ wa?")
```

::: beni.core.morphotactic.distil.distillation.Distiller
    options:
      members:
        - handle_baselines
        - phi_on_texts
        - propose
        - write_checkpoint
        - is_first_create
        - is_scratch
        - checkpoint_id
      show_root_heading: true
      heading_level: 2
      inherited_members: false
      show_if_no_docstring: true

::: beni.core.morphotactic.dabax.resolve_baseline_files
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.morphotactic.dabax.DabaX
    options:
      members:
        - loader
      show_root_heading: true
      heading_level: 2
      inherited_members: false
      show_if_no_docstring: false
