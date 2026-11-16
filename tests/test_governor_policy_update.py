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
