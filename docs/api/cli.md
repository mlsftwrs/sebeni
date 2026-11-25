# `beni.cli`

Console script `sebeni`. Autodoc of the Typer `app` object is omitted (it has
no stable inspect signature and broke the API pages). Commands:

| Command | Purpose |
| --- | --- |
| `sebeni init` | Write `config.yaml` + workdir (`--lang` repeatable) |
| `sebeni train` | SAMPG + policy-update plugin; `--lr`, `--lora-r`, … |
| `sebeni exp` | Multilingual train + eval on `data.source` or the packaged data (`experiment` alias) |
| `sebeni distill` | Distiller only |
| `sebeni eval` | Φ, MER, MCS, UWEC of the trained weights (`--weights`) per language and pooled → `exp/eval.json` |
| `sebeni wordfreq` | Surfaces / lemmas / morphemes / stages |
| `sebeni generate` | Trained-policy decode (`--weights`, `--max-new-tokens`) + R_format / R_lang warnings |
| `sebeni push` | Hub upload of the trained weights (card + eval snapshot for those weights required) |

```bash
sebeni init --lang bam --lang mku -w ./runs/manding-001
sebeni train -c ./runs/manding-001/config.yaml --lr 1e-5 --max-steps 20
sebeni exp -c configs/exp.yaml -w ./runs/exp-001
sebeni --help
sebeni train --help
```

::: beni.cli.main.default_config_yaml
    options:
      show_root_heading: true
      heading_level: 2

!!! note
    `sebeni eval`, `generate`, and `push` load the trained weights from
    `--weights` (default the arm's output dir, `{working_dir}/models`): a LoRA
    adapter is attached to a fresh base model, a full checkpoint is loaded
    directly. Without trained weights `eval` scores the base model with a
    warning and `push` refuses.

See [CLI and config](../cli.md) and [Use cases](../use-cases.md).
