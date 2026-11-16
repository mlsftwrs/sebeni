"""Policy-update gates in the optimizer hook (issues 3, 6, 7)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from beni.core.srl.config import MasterConfig
from beni.core.srl.plugin import AlignmentPlugin


def _wired(model: torch.nn.Module, lr: float = 0.1):
    """A plugin whose step hook wraps a real AdamW on ``model``."""
    plugin = AlignmentPlugin(MasterConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    trainer = SimpleNamespace(model=model, optimizer=optimizer, state=SimpleNamespace(log_history=[]))
    plugin.model = model
    plugin._wire_pre_update_hooks(trainer)
    return plugin, optimizer


def _backward(model):
    model.zero_grad()
    model(torch.ones(1, 2)).sum().backward()


def test_blocked_update_leaves_parameters_unchanged():
    """Issue 7: a format-invalid batch must not move θ (no AdamW momentum / weight decay)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    plugin, optimizer = _wired(model)
    # One allowed step first, so AdamW has momentum that would still move θ.
    plugin.reward_manager.format_scores.extend([0.1])
    _backward(model)
    optimizer.step()
    before = [p.detach().clone() for p in model.parameters()]

    plugin.reward_manager.format_scores.extend([0.0, 0.0])
    _backward(model)
    assert optimizer.step() is None
    for old, new in zip(before, model.parameters()):
        assert torch.equal(old, new)
    assert plugin.blocked_steps == 1
    assert "format_invalid_batch" in plugin.governor.spec.fired


def test_allowed_update_moves_parameters():
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    plugin, optimizer = _wired(model)
    before = [p.detach().clone() for p in model.parameters()]
    plugin.reward_manager.format_scores.extend([0.1, 0.0])
    _backward(model)
    optimizer.step()
    assert any(not torch.equal(o, n) for o, n in zip(before, model.parameters()))
    assert plugin.blocked_steps == 0


# --- issue 3: language gates -------------------------------------------------

import json  # noqa: E402

import pytest  # noqa: E402

from beni.core.compute.rewards import RewardManager  # noqa: E402
from beni.core.safety.governor import SafetyGovernor  # noqa: E402
from beni.core.safety.spec import SafetySpec  # noqa: E402


def _json(lang: str) -> str:
    return json.dumps({"text": "a", "lang": lang, "tokens": [{"surface": "a", "stage": 0, "analyses": []}]})


@pytest.mark.parametrize(
    "completion, expected, result",
    [
        (_json("bam"), "bam", (True, False)),
        (_json("mlq"), "mku", (True, False)),
        (_json("dtm"), "bam", (False, False)),
        (_json("bam,dtm"), "bam", (False, True)),
        ('{"lang": "bam"} {"lang": "dtm"}', "bam", (False, True)),
        ("no json here", "bam", (None, False)),
    ],
)
def test_language_identity(completion, expected, result):
    assert RewardManager.language_identity(completion, expected) == result


def test_reward_lang_scores_mixed_as_zero():
    rm = RewardManager()
    scores = rm.reward_lang([_json("bam"), _json("bam,dtm")], language=["bam", "bam"])
    assert scores[0] > 0 and scores[1] == 0.0


def test_reward_format_observes_language_once_per_batch():
    rm = RewardManager()
    batch = [_json("bam"), _json("dtm")]
    rm.reward_format(batch, language=["bam", "bam"])
    rm.reward_lang(batch, language=["bam", "bam"])
    assert rm.lang_ok == [True, False]
    rm.clear()
    assert rm.lang_ok == [] and rm.mixed_language == []


def _gov(**spec) -> SafetyGovernor:
    return SafetyGovernor(SafetySpec(**spec))


def test_wrong_language_batch_is_refused():
    """A GRPO batch with no completion in the right language is refused."""
    gov = _gov()
    meta = {"signal_source": "samples", "format_scores": [0.1, 0.1], "lang_ok": [False, False], "mixed_language": [False, False]}
    assert gov.apply_pre_update(None, meta) is False
    assert "language_mismatch_batch" in gov.spec.fired


def test_samples_with_one_good_completion_pass():
    gov = _gov()
    meta = {"signal_source": "samples", "format_scores": [0.1, 0.1], "lang_ok": [True, False], "mixed_language": [False, False]}
    assert gov.allow_policy_update(meta) is True


def test_targets_with_one_wrong_language_are_refused():
    gov = _gov()
    meta = {"signal_source": "targets", "format_scores": [0.1, 0.1], "lang_ok": [True, False], "mixed_language": [False, False]}
    assert gov.allow_policy_update(meta) is False


def test_mixed_language_target_is_refused_even_without_require_language():
    gov = _gov(require_language=False)
    meta = {"signal_source": "targets", "format_scores": [0.1], "lang_ok": [False], "mixed_language": [True]}
    assert gov.allow_policy_update(meta) is False
    assert "mixed_language_batch" in gov.spec.fired


def test_mixed_gate_off_allows():
    gov = _gov(require_language=False, refuse_mixed_language=False)
    meta = {"signal_source": "targets", "format_scores": [0.1], "lang_ok": [False], "mixed_language": [True]}
    assert gov.allow_policy_update(meta) is True


def test_undeclared_language_does_not_block():
    gov = _gov()
    meta = {"signal_source": "samples", "format_scores": [0.1], "lang_ok": [None], "mixed_language": [False]}
    assert gov.allow_policy_update(meta) is True


def test_hooked_step_passes_language_signals():
    """The optimizer hook forwards reward_manager.lang_ok to the governor (issue 3)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
    plugin, optimizer = _wired(model)
    before = [p.detach().clone() for p in model.parameters()]
    plugin.reward_manager.reward_format([_json("dtm"), _json("dtm")], language=["bam", "bam"])
    _backward(model)
    assert optimizer.step() is None
    assert all(torch.equal(o, n) for o, n in zip(before, model.parameters()))
    assert "language_mismatch_batch" in plugin.governor.spec.fired
