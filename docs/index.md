<figure markdown="span">
  ![SEBEN!](assets/seben-wordmark.png){ width="360" }
</figure>

# sebeni

<div class="admonition note" markdown>
<p class="admonition-title">session</p>

```
$ whoami
sebeni
$ sebeni exp --preset multi13 --algorithm sft -w ./runs/multi13-sft
distill → train sft → eval MER / MCS / UWEC
```

</div>

<p>
  <span class="typewriter" data-text="$ sebeni exp --preset multi13 --algorithm sft"></span>
</p>

Sebeni is a **morphotactic post-training toolkit** for extremely low-resource
languages. One run is three stages:

1. Distill grammar **G** and dictionary **D** once per language, then freeze them.
2. Train **one** arm — SFT, GRPO, DPO, or APO — against that checkpoint.
3. Score held-out **MER**, **MCS**, and **UWEC**. All three are costs to minimize.

Completions are morphological JSON (`tokens`), not a chatbot.

```bash
pip install "sebeni[train,distil] @ git+https://github.com/mlsftwrs/sebeni.git"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
sebeni --help
```

[Get started](getting-started.md){ .md-button }
[Run an experiment](experiments.md){ .md-button }

Public docs: [https://seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).
Project home: [seben.robotsmali.org](https://seben.robotsmali.org).
Code: [github.com/mlsftwrs/sebeni](https://github.com/mlsftwrs/sebeni).
Hub: [huggingface.co/mlsftwrs](https://huggingface.co/mlsftwrs).

## Objects in the loop

| Symbol | Role |
| --- | --- |
| \(T\) | Dataset of `{text, lang}` rows — not G or D |
| \(G, D\) | Daba grammar and dictionary files (`.gram` / `.dict`), frozen after distill |
| \(\Phi\) | Morphological integrity of the train split given \(G, D\) |
| \(\tau\) | Threshold, default **0.5** |
| \(\theta\) | The SLM policy (**one** model, even when \(T\) is multilingual) |

Maninka group code is **mku** (not `mlq`). In the experiment jsonl only,
`mlq` → `kao`, `hsy` → `mey`, `seq` → `spp`. `bbo` is an outlier.

## Read next

1. [Getting started](getting-started.md) — install, dataset shape, happy path
2. [Experiments](experiments.md) — `sebeni exp` on packaged raw / test data
3. [Use cases](use-cases.md) — ten recipes with your own jsonl
4. [SAMPG](sampg.md) — distill once, train one arm, evaluate
5. [Rewards](rewards.md) — MER, MCS, UWEC, and training rewards
