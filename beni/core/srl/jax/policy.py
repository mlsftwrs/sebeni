"""JAX/Flax policy updates sharing Sebeni's DabaX rewards and safety contract."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from beni.core.srl.grpo.callbacks import log_trackio
from beni.core.srl.plugin import AlignmentPlugin
from beni.data.datasets import build_dabax_reference, rank_group_to_preference

logger = logging.getLogger(__name__)


def _log_metrics(values: dict) -> None:
    step = values.get("step") if isinstance(values, dict) else None
    log_trackio(values, step=step if isinstance(step, int) else None)


def jax_uncertainty_scale(
    predicted_stages, model_logps, ref_logps, beta: float = 0.1
) -> tuple[float, dict]:
    """Framework-neutral U and non-amplifying update scale."""
    stages = [
        stage
        for sentence_stages in (predicted_stages or [])
        for stage in (
            sentence_stages
            if isinstance(sentence_stages, (list, tuple))
            else [sentence_stages]
        )
    ]
    indicators = []
    for stage in stages:
        try:
            indicators.append(float(int(stage) != -1))
        except (TypeError, ValueError):
            indicators.append(1.0)
    indicator = float(np.mean(indicators)) if indicators else 0.0
    model_values = np.asarray(model_logps, dtype=float)
    ref_values = np.asarray(ref_logps, dtype=float)
    u_kl = beta * float(np.mean(model_values - ref_values))
    uncertainty = indicator + u_kl
    return 1.0 / (1.0 + max(uncertainty, 0.0)), {
        "u_indicator": indicator,
        "u_kl": u_kl,
        "uncertainty": uncertainty,
    }


class JaxPolicyPlugin(AlignmentPlugin):
    """Minimal Flax/Optax GRPO and preference loop with torch fallback."""

    name = "jax"

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.params = None
        self.ref_params = None
        self.tx = None
        self.opt_state = None
        self._fallback = None

    def _fallback_to_torch(self, reason: Exception):
        requested = self.config.model.flax_model_name or self.config.model.model_name
        message = (
            f"JAX was requested, but Flax weights could not be loaded from {requested!r}: "
            f"{reason}. Continuing with PyTorch/TRL. To require JAX, set "
            "`model.flax_model_name` (or `model_name`) to a repository exporting Flax weights."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        logger.warning(message)
        print(message)
        self.config.trainer.framework = "torch"
        if self.config.algorithm == "dpo":
            from beni.core.srl.dpo.dpo import SebeniDpo

            cls = SebeniDpo
        elif self.config.algorithm == "apo":
            from beni.core.srl.apo.apo import SebeniApo

            cls = SebeniApo
        else:
            from beni.core.srl.grpo.grpo import SebeniGrpo

            cls = SebeniGrpo
        self._fallback = cls(self.config)
        return self._fallback.load_models()

    def load_models(self, model_name: Optional[str] = None, **kwargs):
        requested = (
            self.config.model.flax_model_name
            or model_name
            or self.config.model.model_name
        )
        try:
            import jax
            import optax
            from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

            self.model = FlaxAutoModelForCausalLM.from_pretrained(requested)
            self.tokenizer = AutoTokenizer.from_pretrained(requested)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.params = self.model.params
            self.ref_params = jax.tree_util.tree_map(
                lambda value: value.copy(), self.params
            )
            training_config = (
                self.config.dpo
                if self.config.algorithm == "dpo"
                else self.config.apo
                if self.config.algorithm == "apo"
                else self.config.trainer
            )
            self.tx = optax.adamw(
                training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
            self.opt_state = self.tx.init(self.params)
            return self
        except Exception as exc:
            return self._fallback_to_torch(exc)

    def _token_logps(self, params, input_ids, attention_mask):
        import jax
        import jax.numpy as jnp

        logits = self.model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            params=params,
            train=True,
        ).logits
        labels = input_ids[:, 1:]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
        return selected * attention_mask[:, 1:]

    def token_probabilities(self, text: str):
        """Per-token π_θ and π_ref as probabilities from Flax, else the torch fallback."""
        import numpy as np

        if self._fallback is not None:
            return self._fallback.token_probabilities(text)
        if self.model is None or self.tokenizer is None:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        import jax.numpy as jnp

        encoded = self.tokenizer(text, return_tensors="np")
        ids = jnp.asarray(encoded["input_ids"])
        mask = jnp.asarray(encoded["attention_mask"])
        if int(ids.shape[-1]) < 2:
            ones = np.asarray([1.0], dtype=float)
            return ones, ones.copy()
        theta = np.asarray(self._token_logps(self.params, ids, mask)[0])
        ref = np.asarray(self._token_logps(self.ref_params, ids, mask)[0])
        return np.exp(theta), np.exp(ref)

    def _grpo_step(self, prompts, completions, advantages, prompt_lengths):
        import jax
        import jax.numpy as jnp
        import optax

        encoded = self.tokenizer(
            [p + c for p, c in zip(prompts, completions)],
            return_tensors="np",
            padding=True,
        )
        ids = jnp.asarray(encoded["input_ids"])
        mask = jnp.asarray(encoded["attention_mask"])
        completion_mask = np.asarray(encoded["attention_mask"][:, 1:], dtype=float)
        for row, length in enumerate(prompt_lengths):
            completion_mask[row, : max(length - 1, 0)] = 0.0
        completion_mask = jnp.asarray(completion_mask)
        advantages = jnp.asarray(advantages)

        ref_logps = self._token_logps(self.ref_params, ids, mask)
        old_logps = self._token_logps(self.params, ids, mask)
        lengths = jnp.maximum(completion_mask.sum(axis=1), 1.0)
        old_seq = (old_logps * completion_mask).sum(axis=1) / lengths
        ref_seq = (ref_logps * completion_mask).sum(axis=1) / lengths

        def loss_fn(params):
            logps = self._token_logps(params, ids, mask)
            seq_logps = (logps * completion_mask).sum(axis=1) / lengths
            ratio = jnp.exp(seq_logps - jax.lax.stop_gradient(old_seq))
            clipped = jnp.clip(ratio, 0.8, 1.2)
            surrogate = jnp.minimum(ratio * advantages, clipped * advantages)
            log_ratio = ref_seq - seq_logps
            kl = jnp.exp(log_ratio) - log_ratio - 1.0
            loss = -surrogate.mean() + self.config.trainer.beta * kl.mean()
            return loss, (seq_logps, ref_seq)

        (loss, (model_logps, ref_values)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.params)
        scale, u = jax_uncertainty_scale(
            self.reward_manager.predicted_stages,
            np.asarray(model_logps),
            np.asarray(ref_values),
            self.config.trainer.beta,
        )
        self._last_uncertainty = u
        grads = jax.tree_util.tree_map(lambda value: value * scale, grads)
        updates, self.opt_state = self.tx.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        return float(loss), u

    def _prompt_text(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.data_loader._completion_ds(
                    [{"text": "", "lang": self.config.data.default_lang}]
                )[0]["prompt"][0]["content"],
            },
            {"role": "user", "content": text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{messages[0]['content']}\n{text}\n"

    def _train_grpo(self, records):
        max_steps = max(1, int(self.config.trainer.max_steps))
        k = max(2, int(self.config.trainer.num_generations))
        history = []
        for step in range(max_steps):
            row = records[step % len(records)]
            text = row.get("text", "")
            lang = row.get("lang") or row.get("language") or self.config.data.default_lang
            reference = row.get("reference") or build_dabax_reference(text, lang)
            prompt = self._prompt_text(text)
            prompt_encoded = self.tokenizer(prompt, return_tensors="np")
            generated = self.model.generate(
                **prompt_encoded,
                params=self.params,
                max_new_tokens=self.config.trainer.max_completion_length,
                do_sample=True,
                temperature=self.config.trainer.temperature,
                num_return_sequences=k,
            ).sequences
            completions = [
                self.tokenizer.decode(
                    seq[prompt_encoded["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                for seq in np.asarray(generated)
            ]
            rewards = self.reward_manager.compute_rewards(
                completions,
                prompts=[{"role": "user", "content": text}] * k,
                languages=[lang] * k,
                reference=[reference] * k,
            )["total"]
            values = np.asarray(rewards, dtype=float)
            advantages = (values - values.mean()) / (values.std() + 1e-8)
            loss, u = self._grpo_step(
                [prompt] * k,
                completions,
                advantages,
                [prompt_encoded["input_ids"].shape[1]] * k,
            )
            history.append({"step": step, "loss": loss, **u})
            _log_metrics(history[-1])
            self.reward_manager.clear()
        return {"history": history}

    def _preference_step(self, prompt, chosen, rejected):
        import jax
        import jax.numpy as jnp
        import optax

        texts = [prompt + chosen, prompt + rejected]
        encoded = self.tokenizer(texts, return_tensors="np", padding=True)
        ids = jnp.asarray(encoded["input_ids"])
        mask = jnp.asarray(encoded["attention_mask"])
        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        completion_mask = np.asarray(encoded["attention_mask"][:, 1:], dtype=float)
        completion_mask[:, : max(prompt_len - 1, 0)] = 0.0
        completion_mask = jnp.asarray(completion_mask)
        ref_token_logps = self._token_logps(self.ref_params, ids, mask)

        def sequence_scores(params):
            token_logps = self._token_logps(params, ids, mask)
            lengths = jnp.maximum(completion_mask.sum(axis=1), 1.0)
            return (token_logps * completion_mask).sum(axis=1) / lengths

        ref_scores = (ref_token_logps * completion_mask).sum(axis=1) / jnp.maximum(
            completion_mask.sum(axis=1), 1.0
        )

        def loss_fn(params):
            scores = sequence_scores(params)
            log_ratios = scores - ref_scores
            beta = (
                self.config.apo.beta
                if self.config.algorithm == "apo"
                else self.config.dpo.beta
            )
            margin = beta * (log_ratios[0] - log_ratios[1])
            if self.config.algorithm == "apo":
                if self.config.apo.loss_type == "apo_down":
                    loss = jax.nn.sigmoid(beta * log_ratios[0]) + (
                        1.0 - jax.nn.sigmoid(margin)
                    )
                else:
                    loss = (
                        1.0 - jax.nn.sigmoid(beta * log_ratios[0])
                    ) + jax.nn.sigmoid(beta * log_ratios[1])
            else:
                loss = -jax.nn.log_sigmoid(margin)
            return loss, (scores, ref_scores)

        (loss, (scores, frozen_scores)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.params)
        chosen_sentences = self.reward_manager.completions_to_sentences([chosen])
        stages = [[token.stage for token in chosen_sentences[0].tokens]]
        beta = self.config.apo.beta if self.config.algorithm == "apo" else self.config.dpo.beta
        scale, u = jax_uncertainty_scale(
            stages, np.asarray(scores), np.asarray(frozen_scores), beta
        )
        self._last_uncertainty = u
        grads = jax.tree_util.tree_map(lambda value: value * scale, grads)
        updates, self.opt_state = self.tx.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        return float(loss), u

    def _train_preference(self, records):
        dataset = rank_group_to_preference(
            records, scheme_prompt=True, languages=self.config.languages()
        )
        max_steps = max(
            1,
            int(
                self.config.apo.max_steps
                if self.config.algorithm == "apo"
                else self.config.dpo.max_steps
            ),
        )
        history = []
        for step in range(max_steps):
            row = dataset[step % len(dataset)]
            prompt_value = row["prompt"]
            if isinstance(prompt_value, list) and hasattr(
                self.tokenizer, "apply_chat_template"
            ):
                prompt = self.tokenizer.apply_chat_template(
                    prompt_value, tokenize=False, add_generation_prompt=True
                )
            elif isinstance(prompt_value, list):
                prompt = "\n".join(message["content"] for message in prompt_value)
            else:
                prompt = str(prompt_value)
            loss, u = self._preference_step(
                prompt, str(row["chosen"]), str(row["rejected"])
            )
            history.append({"step": step, "loss": loss, **u})
            _log_metrics(history[-1])
        return {"history": history}

    def train(self, data, project_name=None, **kwargs):
        if self._fallback is not None:
            return self._fallback.train(data, project_name=project_name, **kwargs)
        if self.model is None:
            self.load_models()
            if self._fallback is not None:
                return self._fallback.train(data, project_name=project_name, **kwargs)
        records = [dict(row) for row in data]
        if not records:
            raise ValueError("JAX PolicyUpdate requires at least one record.")
        self._init_trackio(project_name or self.config.project_name)
        result = (
            self._train_grpo(records)
            if self.config.algorithm == "grpo"
            else self._train_preference(records)
        )
        output_dir = (
            self.config.dpo.output_dir
            if self.config.algorithm == "dpo"
            else self.config.apo.output_dir
            if self.config.algorithm == "apo"
            else self.config.trainer.output_dir
        )
        self.save_model(output_dir)
        self._finish_trackio()
        return result

    def save_model(self, output_dir=None):
        if self._fallback is not None:
            return self._fallback.save_model(output_dir)
        output = Path(output_dir or self.config.trainer.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output, params=self.params)
        self.tokenizer.save_pretrained(output)
        self.write_card_and_snapshot(output)
        return output
