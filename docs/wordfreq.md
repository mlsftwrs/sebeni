# Wordfreq

Morphological counter in the spirit of
[bamfreq](https://mlsftwrs.web.app/app/bamfreq): tokenize and analyze with CLI
mparser / `DabaX.loader` against the **current DabaX checkpoint**
(`baseline_vN` in `{working_dir}/data/baselines/{lang}/`).

Not FastText. FastText / Matryoshka embeddings are a later vectorization track.

## Counts (`{working_dir}/exp/wordfreq/`)

- Surface forms (`token.surface`)
- Lemma / analysis `form` from the best Daba analysis
- Optional morpheme forms from the nested `Morpheme` tree
- Stage histogram (feeds Phi / integrity reporting)
- Stage −1 miss map (feeds algorithmic Distiller)

## Raw-input pipeline

```yaml
data:
  default_lang: bam
  languages: [bam, mku]
wordfreq:
  raw_inputs: ./data/raw   # directory, glob, file, or list
```

Wordfreq does not require `{text, lang}` JSONL. It reads raw text, splits it
into paragraphs/lines, assigns language from `data.languages`, `--lang`, or a
filename stem such as `bam.txt`, and runs DabaX. Unlabelled files use
`data.default_lang`. `data.source` is a compatibility fallback when
`wordfreq.raw_inputs` is unset; otherwise packaged `beni/data/raw` is used.
Relative paths resolve against the config file's directory. Files are read
with `data.encoding` (default `utf-8-sig`).

```
sebeni wordfreq -c config.yaml
sebeni wordfreq -c config.yaml --lang bam --lang mku
```

Writes `{working_dir}/exp/wordfreq/wordfreq.json` plus a subdirectory per
language (`bam/`, `mku/`, …) with surfaces / lemmas / morphemes / stages /
misses. Each language is counted against its latest workdir baseline, falling
back to the packaged baseline. The command does not instantiate an LLM
Distiller and needs no API key.
