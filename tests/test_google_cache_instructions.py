"""Issue 14: the delta / bootstrap instructions reach Gemini with and without a context cache."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("google.genai")

from beni.core.morphotactic.distil.providers.google import GoogleProvider  # noqa: E402


def _provider(cache=None) -> GoogleProvider:
    with patch("beni.core.morphotactic.distil.providers.google.genai.Client"):
        provider = GoogleProvider(api_key="k", language="bam", region="us-central1")
    provider.cache = cache
    provider._client.models.generate_content.return_value = SimpleNamespace(parsed={})
    return provider


def test_construction_creates_no_cache():
    with patch("beni.core.morphotactic.distil.providers.google.genai.Client") as client:
        provider = GoogleProvider(api_key="k", language="bam", region="us-central1")
    assert provider.cache is None
    client.return_value.caches.create.assert_not_called()


def test_uncached_request_carries_system_instruction():
    provider = _provider()
    provider.generate(prompt="misses", sys_instruct="APPLY [ADD] DELTAS")
    kwargs = provider._client.models.generate_content.call_args.kwargs
    assert kwargs["config"].system_instruction == "APPLY [ADD] DELTAS"
    assert kwargs["contents"] == "misses"


def test_cached_request_carries_instructions_in_contents():
    provider = _provider(cache=SimpleNamespace(name="cachedContents/1"))
    provider.generate(prompt="misses", sys_instruct="APPLY [ADD] DELTAS")
    kwargs = provider._client.models.generate_content.call_args.kwargs
    assert kwargs["config"].cached_content == "cachedContents/1"
    assert kwargs["config"].system_instruction is None
    assert kwargs["contents"][0].endswith("APPLY [ADD] DELTAS")
    assert kwargs["contents"][1] == "misses"
