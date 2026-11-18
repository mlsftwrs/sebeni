"""Issue 21: the device follows the active arm's block, and CPU fallback updates all blocks."""

from unittest.mock import MagicMock, patch

import pytest

from beni.core.srl.config import MasterConfig
from beni.core.srl.plugin import AlignmentPlugin, _resolve_device


def _config(algorithm: str, **use_cpu) -> MasterConfig:
    mc = MasterConfig()
    mc.algorithm = algorithm
    mc.model.load_in_4bit = False
    mc.model.use_peft = False
    for block in ("trainer", "sft", "dpo", "apo"):
        getattr(mc, block).use_cpu = bool(use_cpu.get(block, False))
    return mc


@pytest.mark.parametrize("algorithm, block", [("sft", "sft"), ("grpo", "trainer"), ("dpo", "dpo"), ("apo", "apo")])
def test_active_block_use_cpu_selects_cpu_with_cuda(algorithm, block):
    mc = _config(algorithm, **{block: True})
    with patch("beni.core.srl.plugin.torch.cuda.is_available", return_value=True):
        assert _resolve_device(mc) == "cpu"


def test_inactive_block_use_cpu_is_ignored():
    mc = _config("sft", trainer=True)
    with patch("beni.core.srl.plugin.torch.cuda.is_available", return_value=True):
        assert _resolve_device(mc) == "auto"


@patch("beni.core.srl.plugin.AutoTokenizer.from_pretrained")
@patch("beni.core.srl.plugin.AutoModelForCausalLM.from_pretrained")
def test_sft_use_cpu_loads_on_cpu(mock_model, mock_tok):
    mock_model.return_value = MagicMock()
    mock_tok.return_value = MagicMock(pad_token="<pad>", eos_token="<eos>", chat_template="x")
    mc = _config("sft", sft=True)
    with patch("beni.core.srl.plugin.torch.cuda.is_available", return_value=True):
        AlignmentPlugin(mc).load_models()
    assert mock_model.call_args.kwargs["device_map"] == "cpu"


@patch("beni.core.srl.plugin.AutoTokenizer.from_pretrained")
@patch("beni.core.srl.plugin.AutoModelForCausalLM.from_pretrained")
def test_cpu_fallback_sets_all_blocks(mock_model, mock_tok):
    mock_model.return_value = MagicMock()
    mock_tok.return_value = MagicMock(pad_token="<pad>", eos_token="<eos>", chat_template="x")
    mc = _config("sft")
    with patch("beni.core.srl.plugin.torch.cuda.is_available", return_value=False):
        AlignmentPlugin(mc).load_models()
    assert mock_model.call_args.kwargs["device_map"] == "cpu"
    assert all(getattr(mc, block).use_cpu for block in ("trainer", "sft", "dpo", "apo"))
