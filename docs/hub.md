# Hub model cards

Every `save_model` writes `README.md` and `safety_snapshot.json` into
`output_dir` (`beni.core.hub.model_card`). That training snapshot has no eval
metrics, so it does not clear the release gate on its own: `sebeni eval`
rewrites `safety_snapshot.json` next to the weights it scored, with Φ, MER /
MCS and a `weights_sha256` of those weight files. Push never writes the card or
snapshot itself; it only checks them.

The card includes HF YAML frontmatter, intended use (ELRL morphological
generation — not a chatbot), training recipe (algorithm, rewards, gates that
fired), eval table (Phi, MER, MCS, UWEC, format, R_lang), languages/groups (MKU not MLQ),
baseline **grammar/dictionary** checkpoint id, hyperparameter snapshot, and links to
[seben.robotsmali.org](https://seben.robotsmali.org) and the docs at
[mlsftwrs.github.io/sebeni](https://mlsftwrs.github.io/sebeni/).

Trained policies are Hub model repos. **Source code** stays on GitHub
([mlsftwrs/sebeni](https://github.com/mlsftwrs/sebeni)).

## Push

Train, evaluate the trained weights, then push them:

```bash
huggingface-cli login
sebeni train -c config.yaml
sebeni eval -c config.yaml                     # writes the snapshot next to the weights
sebeni push -c config.yaml --repo-id mlsftwrs/sebeni-manding-grpo --write-card
```

`eval` and `push` load the **trained** weights: `--weights PATH|REPO` (default:
the arm's output dir, `{working_dir}/models`). A LoRA adapter is attached to a
fresh base model, so a LoRA run pushes the adapter; a full checkpoint is pushed
as a full model. A Hub repo id resolves through the local HF cache. `push`
refuses without trained weights.

The push gate needs `README.md` and an eval `safety_snapshot.json` **next to
the pushed weights**, and the snapshot must have been written by `sebeni eval`
for exactly those weight files (see [Safety](safety.md)). `--write-card`
re-renders `README.md` from that eval snapshot. The push then uploads exactly
the checked folder (`HfApi.upload_folder`; `checkpoint-*` directories are
excluded), not whatever model is in memory.

YAML: `trainer.hub_model_id`, `push_to_hub`, `hub_private_repo`. Token via
`HF_TOKEN` or `huggingface-cli login` (write access on org `mlsftwrs`).

```yaml
trainer:
  push_to_hub: true
  hub_model_id: mlsftwrs/sebeni-manding-grpo
  hub_private_repo: true   # set false for a public model
  # hub_token: null        # prefer HF_TOKEN / huggingface-cli login
```

`<arm>.push_to_hub: true` tries a push at the end of training. Training has no
eval snapshot yet, so the gate normally refuses; the refusal is reported and
the trained run is kept. The HF Trainer never pushes checkpoints by itself. Use
`sebeni eval` then `sebeni push` for a release.

## Transfer into Hugging Face org `mlsftwrs`

The Hub org holds **models / datasets / Spaces**. It does not replace GitHub
for this source tree (Pages still need `mlsftwrs/sebeni` on GitHub). Hugging
Face does **not** move a GitHub repository.

1. Create [huggingface.co/organizations/new](https://huggingface.co/organizations/new) named `mlsftwrs`.
2. Invite pushers (**write**) and people who will move repos (**admin**).
3. Issue a token with write (and org) scope; `huggingface-cli login`. The token’s
   user must be a member of `mlsftwrs`.
4. **Existing** Hub repos: owner + org admin → **Settings → Transfer repository** → `mlsftwrs`.
5. **This git tree**: nothing to transfer on the Hub until `sebeni push` creates a model repo.
6. Then set `hub_model_id: mlsftwrs/<name>`. Re-add collaborators via the org if
   they were individual users on the old repo. Spaces that referenced the old
   model id need their env/secrets updated.

Optional Hub extras:

- Organization card at `https://huggingface.co/mlsftwrs`
- Dataset repo `mlsftwrs/<name>` if you publish the (text, lang) corpus
- Keep models private (`hub_private_repo: true`) until the card/snapshot look right
- Do not vendor GPL Daba sources into a Hub model repo; the card already credits
  [maslinych/daba](https://github.com/maslinych/daba)
