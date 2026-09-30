import re
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from beni.core.compute.metrics import MorphologyScorer
from beni.core.compute import Sentence, Token, Analysis, Morpheme
from beni.core.morphotactic import dabax
from beni.core.srl.config import RewardConfig
from beni.utils import config as cfg


class RewardManager:
    """
    Reward manager implementing R_morph, R_rule, R_format, and user-customized rewards.

    All reward functions follow the TRL GRPO convention:
        f(completions, language=..., **kwargs) -> List[float]

    GRPOTrainer automatically injects extra dataset columns (e.g. "language")
    as keyword arguments, so the "language" column from format_dataset()
    flows into compute_rewards / the reward functions without any extra plumbing.
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        scorer: Optional[MorphologyScorer] = None,
        working_dir: Optional[str] = None,
    ):
        self.config = reward_config or RewardConfig()
        self.scorer = scorer or MorphologyScorer()
        self.working_dir = Path(working_dir) if working_dir else cfg.get_workdir().root
        self.format_scores: List[float] = []
        self.total_reward: float = 0.0
        self.predicted_stages: List[List[Any]] = []
        self.custom_rewards: Dict[str, Tuple[Callable, float]] = {}
        self.recent_rewards: Dict[str, float] = {}

    def register_custom_reward(self, name: str, reward_fn: Callable, weight: float = 1.0) -> None:
        """Register a custom reward function with a given weight."""
        self.custom_rewards[name] = (reward_fn, weight)

    def get_reward_functions(self) -> List[Callable]:
        """Return list of active reward functions wrapped to apply config weights."""
        funcs = []
        if self.config.enable_format_reward:
            funcs.append(self.reward_format)
        if self.config.enable_morph_reward:
            funcs.append(self.reward_morph)
        if self.config.enable_rule_reward:
            funcs.append(self.reward_rule)
        if getattr(self.config, "enable_lang_reward", False):
            funcs.append(self.reward_lang)
        for name, (fn, w) in self.custom_rewards.items():
            def make_wrapped(reward_func=fn, weight=w):
                def wrapped(*args, **kwargs):
                    scores = reward_func(*args, **kwargs)
                    return [s * weight for s in scores]
                wrapped.__name__ = reward_func.__name__
                return wrapped
            funcs.append(make_wrapped())
        return funcs

    def clear(self) -> None:
        self.format_scores.clear()
        self.total_reward = 0.0
        self.predicted_stages.clear()
        self.recent_rewards.clear()

    @staticmethod
    def _as_list(value: Any) -> list:
        return value if isinstance(value, list) else []

    @staticmethod
    def dict_to_morpheme(d: dict) -> Morpheme:
        children = None
        if isinstance(d, dict) and "morphemes" in d:
            children = [
                RewardManager.dict_to_morpheme(child)
                for child in RewardManager._as_list(d.get("morphemes"))
                if isinstance(child, dict)
            ]

        payload = d if isinstance(d, dict) else {}
        return Morpheme(
            form=payload.get("form") or "",
            ps=RewardManager._as_list(payload.get("ps")),
            gloss=payload.get("gloss") or "",
            morphemes=children,
        )

    @staticmethod
    def parse_json_to_sentence(json_data: dict, text: str = "") -> Sentence:
        payload = json_data if isinstance(json_data, dict) else {}
        tokens = []
        for t_data in RewardManager._as_list(payload.get("tokens")):
            if not isinstance(t_data, dict):
                continue
            analyses = []
            for a_data in RewardManager._as_list(t_data.get("analyses")):
                if not isinstance(a_data, dict):
                    continue
                morphemes = [
                    RewardManager.dict_to_morpheme(m)
                    for m in RewardManager._as_list(a_data.get("morphemes"))
                    if isinstance(m, dict)
                ]
                analyses.append(Analysis(
                    form=a_data.get("form") or "",
                    ps=RewardManager._as_list(a_data.get("ps")),
                    gloss=a_data.get("gloss") or "",
                    morphemes=morphemes,
                ))
            tokens.append(Token(
                surface=t_data.get("surface") or "",
                stage=t_data.get("stage") if t_data.get("stage") is not None else -1,
                analyses=analyses,
            ))
        return Sentence(
            text=text or payload.get("text") or "",
            lang=payload.get("lang") or "bam",
            tokens=tokens,
        )

    def _dabax(self, lang: str):
        return dabax.get_dabax(
            lang,
            process=True,
            runtime_dir=self.working_dir / "runtime",
            working_dir=self.working_dir,
        )

    @staticmethod
    def _lemma_overlap(predicted: Sentence, reference: Sentence) -> float:
        if not reference.tokens:
            return 0.0
        matches = 0
        for pred, ref in zip(predicted.tokens, reference.tokens):
            pred_form = pred.analyses[0].form if pred.analyses else ""
            ref_form = ref.analyses[0].form if ref.analyses else ""
            matches += int(bool(ref_form) and pred_form == ref_form)
        return matches / max(len(reference.tokens), len(predicted.tokens))

    @staticmethod
    def completion_text(completion: Any) -> str:
        """Unwrap TRL conversational completions to the assistant string.

        GRPO 1.x passes ``[[{"role": "assistant", "content": "..."}]]`` when the
        dataset prompt is a chat. Older TRL passed the decoded string.
        """
        if completion is None:
            return ""
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            return str(completion.get("content") or completion.get("text") or "")
        if isinstance(completion, (list, tuple)):
            if not completion:
                return ""
            for item in reversed(completion):
                if isinstance(item, dict) and str(item.get("role", "")).lower() == "assistant":
                    return str(item.get("content") or "")
            last = completion[-1]
            if isinstance(last, dict):
                return str(last.get("content") or last.get("text") or "")
            if isinstance(last, str):
                return last
            return RewardManager.completion_text(last)
        return str(completion)

    @staticmethod
    def extract_json(text: Any) -> dict:
        raw_text = RewardManager.completion_text(text).strip()
        if not raw_text:
            return {}

        # 1. Direct parse attempt
        try:
            res = json.loads(raw_text)
            if isinstance(res, dict):
                return res
            if isinstance(res, str):
                nested = RewardManager.extract_json(res)
                if nested:
                    return nested
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Extract from markdown code fences: ```json ... ``` or ``` ... ```
        fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        for match in fence_pattern.finditer(raw_text):
            block = match.group(1).strip()
            try:
                res = json.loads(block)
                if isinstance(res, dict):
                    return res
            except (json.JSONDecodeError, TypeError):
                pass

        # 3. Balanced brace finder to extract valid JSON objects even with surrounding text
        start_idx = 0
        while True:
            open_pos = raw_text.find('{', start_idx)
            if open_pos == -1:
                break
            depth = 0
            in_str = False
            escape = False
            close_pos = -1
            for i in range(open_pos, len(raw_text)):
                c = raw_text[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            close_pos = i
                            break
            if close_pos != -1:
                candidate = raw_text[open_pos:close_pos + 1].strip()
                try:
                    res = json.loads(candidate)
                    if isinstance(res, dict):
                        return res
                except (json.JSONDecodeError, TypeError):
                    pass
                start_idx = open_pos + 1
            else:
                break

        # 4. Fallback regex greedy attempt
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                res = json.loads(match.group(0))
                if isinstance(res, dict):
                    return res
            except (json.JSONDecodeError, TypeError):
                pass

        return {}

    def completions_to_sentences(self, completions: List[str]) -> List[Sentence]:
        """Shared parse step: raw model completions -> Sentence objects."""
        sentences = []
        for completion in completions:
            json_data = self.extract_json(completion)
            if json_data:
                sentences.append(self.parse_json_to_sentence(json_data, json_data.get("text", "")))
            else:
                sentences.append(Sentence(text="", tokens=[]))
        return sentences

    def reward_format(self, 
        completions: List[str], 
        language: Optional[List[str]] = None, **kwargs) -> List[float]:
        """R_format: Valid JSON structure with a 'tokens' list."""

        scores = []
        
        for text in completions:
            parsed = self.extract_json(text)
            valid = bool(parsed and isinstance(parsed.get("tokens"), list))
            score = self.config.format_weight if valid else 0.0
            scores.append(score)
            self.format_scores.append(score)
            self.total_reward += score

        if scores:
            self.recent_rewards["reward_format"] = float(sum(scores) / len(scores))
        return scores

    def reward_morph(self, 
        completions: List[str], 
        language: Optional[List[str]] = None,
        reference=None,
        prompts=None,
        **kwargs,
    ) -> List[float]:
        """R_morph: annotation quality against DabaX ideal JSON y*."""

        scores = []
        predicted = self.completions_to_sentences(completions)
        references = self._parse_references(prompts or [{}] * len(completions), reference)
        for sent, ref_sent in zip(predicted, references):
            if not sent.tokens or not ref_sent.tokens:
                scores.append(0.0)
                continue
            mcs = self.scorer.mcs(sent, ref_sent)
            mer = self.scorer.mer_sentence(ref_sent, sent, agg="micro")
            mer_quality = 0.0 if not math.isfinite(mer) else 1.0 - min(max(mer, 0.0), 1.0)
            lemma = self._lemma_overlap(sent, ref_sent)
            score = self.config.morph_weight * ((mcs + mer_quality + lemma) / 3.0)
            scores.append(score)
            self.total_reward += score
            self.predicted_stages.append([t.stage for t in sent.tokens])

        if scores:
            self.recent_rewards["reward_morph"] = float(sum(scores) / len(scores))
        return scores

    def reward_rule(
        self, prompts: List[Dict[str, Any]], completions: List[str], 
        language: Optional[List[str]] = None, reference=None, **kwargs) -> List[float]:

        """R_rule: POS/valence adherence via Phi & MCS against the reference."""
        predicted = self.completions_to_sentences(completions)
        reference_sentences = self._parse_references(prompts, reference=reference)
        langs = language or [""] * len(completions)

        scores = []
        for i, (pred_sent, ref_sent, lang) in enumerate(zip(predicted, reference_sentences, langs)):
            try:
                daba_x = self._dabax(lang)
                pred_loaded = daba_x.loader(pred_sent.text) if pred_sent.text else []
            except Exception:
                pred_loaded = []
            pred_sent = pred_loaded[0] if pred_loaded else pred_sent

            phi_score = self.scorer.phi(pred_sent)
            lexical_score = self._lemma_overlap(pred_sent, ref_sent)

            score = self.config.rule_weight * 0.5 * (phi_score + lexical_score)
            scores.append(score)

            self.total_reward += score
            self.predicted_stages.append([t.stage for t in pred_sent.tokens])

        if scores:
            self.recent_rewards["reward_rule"] = float(sum(scores) / len(scores))
        return scores

    @staticmethod
    def _parse_references(prompts: List[Dict[str, Any]], reference=None) -> List[Sentence]:
        reference_sentences = []

        if reference is not None:
            for ref in reference:
                if isinstance(ref, dict) and RewardManager._as_list(ref.get("tokens")):
                    reference_sentences.append(
                        RewardManager.parse_json_to_sentence(ref, ref.get("text", "")))
                elif isinstance(ref, str) and ref:
                    parsed = RewardManager.extract_json(ref)
                    if parsed:
                        reference_sentences.append(
                            RewardManager.parse_json_to_sentence(
                                parsed, parsed.get("text", "")
                            ))
                    else:
                        reference_sentences.append(Sentence(text="", tokens=[]))
                else:
                    reference_sentences.append(Sentence(text="", tokens=[]))
            return reference_sentences

        for prompt in prompts:
            if isinstance(prompt, dict) and ('role' in prompt):
                text = prompt.get("content", "")
                ref_json = RewardManager.extract_json(text)
                if ref_json:
                    reference_sentences.append(RewardManager.parse_json_to_sentence(ref_json, text))
                else:
                    reference_sentences.append(Sentence(text="", tokens=[]))
            else:
                reference_sentences.append(Sentence(text="", tokens=[]))
        
        return reference_sentences

    def reward_lang(
        self,
        completions: List[str],
        language: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """R_lang: language identity. Mixed or mismatched JSON ``lang`` scores 0."""
        from beni.core.language import Language

        scores = []
        langs = language or [""] * len(completions)
        weight = getattr(self.config, "lang_weight", 0.1)

        for text, lang in zip(completions, langs):
            parsed = self.extract_json(text)
            expected = Language.from_code(lang).group_code if lang else ""
            got = str(parsed.get("lang", "") or "").strip().lower()
            if not parsed or not expected or not got:
                score = 0.0
            else:
                got_group = Language.from_code(got).group_code
                score = weight if got_group == expected else 0.0
            scores.append(score)
            self.total_reward += score

        if scores:
            self.recent_rewards["reward_lang"] = float(sum(scores) / len(scores))
        return scores

    def compute_rewards(
        self,
        completions: List[str],
        prompts: List[Dict[str, Any]],
        languages: Optional[List[str]] = None,
        reference=None,
    ) -> Dict[str, List[float]]:
        """Compute all rewards for a batch of completions."""
        format_scores = self.reward_format(completions, language=languages)
        morph_scores = self.reward_morph(
            completions, language=languages, reference=reference, prompts=prompts
        )
        rule_scores = self.reward_rule(
            completions, prompts=prompts, language=languages, reference=reference
        )
        lang_scores = self.reward_lang(completions, language=languages) if getattr(
            self.config, "enable_lang_reward", False) else [0.0] * len(completions)

        total_scores = [
            f + m + r + lg
            for f, m, r, lg in zip(format_scores, morph_scores, rule_scores, lang_scores)
        ]
        if total_scores:
            self.recent_rewards["reward_total"] = float(sum(total_scores) / len(total_scores))

        return {
            "format": format_scores,
            "morph": morph_scores,
            "rule": rule_scores,
            "lang": lang_scores,
            "total": total_scores,
        }
