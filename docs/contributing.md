# Contributing / code style

Match the existing dialect: dataclasses with type hints, TRL-wrapping classes
(`SebeniGrpo`, `RewardManager`, `Distiller`, `DabaX`), numpy-style or short class
docstrings, `beni.*` imports.

- Do not introduce a second config library or rename public APIs.
- Public functions and protocols get numpy-style docstrings (mkdocstrings).
- When editing a file, fix only the touched region. No repo-wide reformat.
- **Ruff**: `format` + `check`, Python 3.10, double quotes, line length 100.
  Run it on new/changed files. Optional `[dev]` extra.

Install for docs / tests:

```bash
pip install -e ".[docs,dev]"
# or: pip install "sebeni[docs,dev] @ git+https://github.com/mlsftwrs/sebeni.git"
ruff format beni tests
ruff check beni tests
pytest
mkdocs serve
```

The docs site uses the terminal MkDocs theme (`docs/stylesheets/terminal.css`):
JetBrains Mono, `cli-dark` / `cli-light` palettes, and a typewriter prompt on
the home page. Keep MathJax (`javascripts/mathjax.js`) for the metric pages.
Public URL: [seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).
