"""One policy prompt for training, evaluation, and generation.

Every arm trains on ``system = SRLGrpoPrompt(languages).prompt()`` plus the
sentence as the user turn. GRPO / DPO / APO hand TRL the chat messages
(:func:`policy_messages`); SFT and decoding render them with the tokenizer's
chat template (:func:`build_policy_prompt`, :func:`build_policy_example`), so
the evaluated policy sees the same input it was trained on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# End-of-turn markers cut from decoded text. A ChatML fallback template on a base
# tokenizer renders ``<|im_end|>`` as plain text, so ``skip_special_tokens`` keeps it.
END_OF_TURN_MARKERS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>")


def system_prompt(languages: Optional[Sequence[str]] = None) -> str:
    """The SRL system prompt naming the run's languages (scope aliases such as ``multi13`` expanded)."""
    from beni.core.language import resolve_scope
    from beni.core.srl.config import SRLGrpoPrompt

    codes = list(resolve_scope(list(languages)).languages) if languages else None
    return SRLGrpoPrompt(languages=codes).prompt()


def policy_messages(text: str, languages: Optional[Sequence[str]] = None) -> List[Dict[str, str]]:
    """Chat messages for one sentence: the system prompt, then the sentence as the user turn."""
    return [
        {"role": "system", "content": system_prompt(languages)},
        {"role": "user", "content": str(text)},
    ]


def _has_template(tokenizer: Any) -> bool:
    return bool(tokenizer is not None and getattr(tokenizer, "chat_template", None))


def build_policy_prompt(
    text: str,
    tokenizer: Any = None,
    languages: Optional[Sequence[str]] = None,
) -> str:
    """Render the policy prompt, ending where the assistant's answer starts.

    Uses ``tokenizer.apply_chat_template`` when the tokenizer has a template
    (``load_models`` sets a ChatML fallback when the Hub tokenizer has none);
    otherwise a plain ``System / User / Assistant`` layout.
    """
    messages = policy_messages(text, languages)
    if _has_template(tokenizer):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{messages[0]['content']}\n\nUser: {text}\nAssistant: "


def build_policy_example(
    text: str,
    completion: str,
    tokenizer: Any = None,
    languages: Optional[Sequence[str]] = None,
) -> str:
    """Full SFT training text: see :func:`build_policy_example_parts`."""
    prompt, answer = build_policy_example_parts(text, completion, tokenizer, languages)
    return prompt + answer


def build_policy_example_parts(
    text: str,
    completion: str,
    tokenizer: Any = None,
    languages: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    """SFT ``(prompt, answer)``: exactly :func:`build_policy_prompt`, then the answer and end of turn.

    The prompt is an exact prefix, so templates that inject text before the
    final assistant message (e.g. empty think blocks) cannot teach the policy
    output that decoding never prompts for. A leading BOS rendered by the
    template is dropped, because the SFT tokenizer adds its own.
    """
    prompt = build_policy_prompt(text, tokenizer, languages)
    suffix = ""
    if _has_template(tokenizer):
        messages = policy_messages(text, languages) + [{"role": "assistant", "content": completion}]
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        index = full.rfind(completion)
        if index != -1:
            suffix = full[index + len(completion):]
        bos = getattr(tokenizer, "bos_token", None)
        if isinstance(bos, str) and bos and prompt.startswith(bos):
            prompt = prompt[len(bos):]
    return prompt, completion + suffix


def end_of_turn_ids(tokenizer: Any) -> List[int]:
    """Token ids that end the assistant turn: eos plus any end-of-turn marker in the vocabulary."""
    ids: List[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int):
        ids.append(eos_id)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for marker in END_OF_TURN_MARKERS:
        token_id = vocab.get(marker) if isinstance(vocab, dict) else None
        if isinstance(token_id, int) and token_id != unk_id and token_id not in ids:
            ids.append(token_id)
    return ids


def strip_end_of_turn(completion: str, tokenizer: Any = None) -> str:
    """Cut decoded text at the first end-of-turn marker (or the tokenizer's eos)."""
    markers = list(END_OF_TURN_MARKERS)
    eos = getattr(tokenizer, "eos_token", None)
    if isinstance(eos, str) and eos:
        markers.append(eos)
    cut = len(completion)
    for marker in markers:
        index = completion.find(marker)
        if index != -1:
            cut = min(cut, index)
    return completion[:cut].strip()
