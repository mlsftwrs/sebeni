
"""
Morphological Analysis Scoring Metrics
======================================

Unified module for evaluating morphological decomposition quality.
Includes MER, MCS, Uncertainty, and Phi scores.

Metrics:
    - MER  : micro morpheme edit rate (S_m + D_m + I_m) / N_m
    - MCS  : match fraction in rewards; mismatch cost in held-out reports
    - UWEC : I(stage != -1) + beta * |log((pi_theta + eps) / (pi_ref + eps))|
    - Phi  : corpus-level morphological integrity score
"""

from __future__ import annotations
import numpy as np
import warnings
from typing import Union, List, Dict
from beni.core.compute.helpers import mer_micro, to_morpheme_list
from beni.core.compute.uwec import uwec
from beni.core import Token, Sentence


class MorphologyScorer:
    """
    Unified scorer for morphological analysis evaluation.
    """

    def __init__(self, beta: float = 0.1, eps: float = 1e-8, empr: float = 0.5):
        self.beta = beta
        self.eps = eps
        self.empr = empr

    def phi(self, sentence: Union[Sentence, List[Token]]) -> float:
        """
        Compute Phi (Morphological Integrity Score)
        Score Structure:
            - stage < 0 : 0.0 Non-Language tokens
            - 0 <= stage < 6: 1.0  (analyzed)
            - stage == 6: empr (Empirical / Emprunt/borrowing foreign token)
            - stage > 6: eps (fallback value)
            - non-numeric parser labels: 1.0
        """
        from beni.core.compute.helpers import stage_to_phi
        tokens = sentence.tokens if isinstance(sentence, Sentence) else sentence
        
        if(not tokens):
            return 0.0
        
        scores = [stage_to_phi(t.stage) for t in tokens]
        return sum(scores) / len(scores)

    def phi_corpus(self, sentences: List[Sentence]) -> Dict[str, float]:
        """ Compute phi for list of texts """
        mapping = {sent.text: self.phi(sent) for sent in sentences}
        return {
            'scores': mapping,
            'avg': float(np.mean(list(mapping.values()))) if mapping else 0.0
        }

    def mer(self, ref_tokens, hyp_tokens) -> float:
        """
        Compute Morpheme Error Rate.

        MER = (S_m + D_m + I_m) / N_m

        Parameters
        ----------
        ref_tokens : Token or list of str
            Reference morpheme sequence (or a Token whose best analysis is used).
        hyp_tokens : Token or list of str
            Hypothesis morpheme sequence.

        Returns
        -------
        float
            Edit-distance rate. 0 when both sequences are empty; inf when
            the reference is empty and the hypothesis is not.
        """
        ref = to_morpheme_list(ref_tokens)
        hyp = to_morpheme_list(hyp_tokens)
        return float(mer_micro(ref, hyp))

    def mer_sentence(self, ref_sentence: Sentence, hyp_sentence: Sentence, agg: str = 'mean') -> float:
        """
        agg: mean vs sum / micro
        TODO: NAIVE & Requires review
        """
        if agg == 'micro':
            refs = [
                form
                for token in ref_sentence.tokens
                for form in token.best_morphemes()
            ]
            hyps = [
                form
                for token in hyp_sentence.tokens
                for form in token.best_morphemes()
            ]
            return self.mer(refs, hyps)


        scores = [self.mer(rt, ht) for rt, ht in zip(ref_sentence.tokens, hyp_sentence.tokens)]

        if agg == 'mean':
            return np.mean(scores) if scores else 0.0
        return np.sum(scores) if scores else 0.0

    def mcs(self, predicted: Union[Sentence, List[Token]], reference: Union[Sentence, List[Token]]) -> float:
        """
        Compute Morphosyntatic Consistency Score
        MCS = (1/|T|) * Σ I(Stage_pred(w) == Stage_ref(w))
        """
        pred_tokens = predicted.tokens if isinstance(predicted, Sentence) else predicted
        ref_tokens = reference.tokens if isinstance(reference, Sentence) else reference
        
        if len(pred_tokens) != len(ref_tokens):
            # FIXME: Token capping enfored to min length(prd, ref)
            warnings.warn(f"Token count mismatch: {len(pred_tokens)} vs {len(ref_tokens)}")
        
        if not ref_tokens:
            return 0.0

        correct = sum(
            1 for pt, rt in zip(pred_tokens, ref_tokens)
            if str(pt.stage) == str(rt.stage))

        return correct / max(len(ref_tokens), len(pred_tokens))

    def mcs_corpus(self, 
                   predicted: List[Sentence], 
                   reference: List[Sentence]) -> float:
        """MCS over an entire corpus [mu-averaged]."""
        all_correct = 0
        all_total = 0

        for ps, rs in zip(predicted, reference):
            ptokens = ps.tokens if isinstance(ps, Sentence) else ps
            rtokens = rs.tokens if isinstance(rs, Sentence) else rs
            all_correct += sum(
                1 for p, r in zip(ptokens, rtokens)
                if str(p.stage) == str(r.stage)
            )
            all_total += len(ptokens)

        return all_correct / all_total if all_total > 0 else 0.0


    def uncertainty_weighted_cost_error(self, 
                    token: Token, 
                    model_prob: float, 
                    ref_prob: float) -> float:
        """Token UWEC: indicator plus absolute log-ratio. Evaluation cost."""
        _, mean = uwec(
            [token.stage],
            [model_prob],
            [ref_prob],
            beta=self.beta,
            eps=self.eps,
        )
        return float(mean)
    
    def uncertainty_weighted_cost_error_sentence(self, 
                    sentence: Sentence, 
                    model_probs: List[float], 
                    ref_probs: List[float]) -> float:
        """Sentence UWEC as the token mean of U(w_i, o)."""
        stages = [token.stage for token in (sentence.tokens or [])]
        _, mean = uwec(
            stages,
            model_probs,
            ref_probs,
            beta=self.beta,
            eps=self.eps,
        )
        return float(mean)

    def evaluate(self, 
                 predicted: Union[Sentence, List[Token]], 
                 reference: Union[Sentence, List[Token]] = None):
        """
        Compute all metrics for a single sentence or a list of sentences.
        """
        results = {
            "mer": self.mer_sentence(reference, predicted),
            "mcs": self.mcs(predicted, reference),
            "phi": self.phi(predicted)
        }

        if isinstance(predicted, list) and isinstance(reference, list):
            results['mer'] = self.mer_sentence(reference, predicted, agg='mean')
            results['mcs'] = self.mcs_corpus(predicted, reference)
            results['phi'] = self.phi_corpus(predicted)['avg']

        return results
