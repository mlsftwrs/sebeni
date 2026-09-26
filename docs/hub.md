# Hub model cards

Every `save_model` / `push_to_hub` writes `README.md` into `output_dir` before
upload (`beni.core.hub.model_card`). `push_to_hub` refuses to run if the card or
`safety_snapshot.json` is missing.

The card includes HF YAML frontmatter, intended use (ELRL morphological
generation — not a chatbot), training recipe (algorithm, rewards, gates that
fired), eval table (Phi, MER, MCS, UWEC, format, R_lang), languages/groups (MKU not MLQ),
baseline **grammar/dictionary** checkpoint id, hyperparameter snapshot, and links to
[seben.robotsmali.org](https://seben.robotsmali.org) and
[seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).

Trained policies are Hub model repos. **Source code** stays on GitHub
([mlsftwrs/sebeni](https://github.com/mlsftwrs/sebeni)).

## Push

```bash
huggingface-cli login
sebeni push -c config.yaml --repo-id mlsftwrs/sebeni-manding-grpo
```

YAML: `trainer.hub_model_id`, `push_to_hub`, `hub_private_repo`. Token via
`HF_TOKEN` or `huggingface-cli login` (write access on org `mlsftwrs`).

```yaml
trainer:
  push_to_hub: true
  hub_model_id: mlsftwrs/sebeni-manding-grpo
  hub_private_repo: true   # set false for a public model
  # hub_token: null        # prefer HF_TOKEN / huggingface-cli login
```

`sebeni push` refuses to run if `README.md` (card) or `safety_snapshot.json`
is missing.

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
