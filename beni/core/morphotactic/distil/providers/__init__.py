import inspect
from typing import Any, Dict, Type

from beni.core.morphotactic.distil.providers import base


PROVIDER_REGISTRY = {
    "google": "beni.core.morphotactic.distil.providers.google:GoogleProvider",
    "gemini": "beni.core.morphotactic.distil.providers.google:GoogleProvider",
    "openai": "beni.core.morphotactic.distil.providers.openai_compat:OpenAIProvider",
    "groq": "beni.core.morphotactic.distil.providers.openai_compat:GroqProvider",
    "together": "beni.core.morphotactic.distil.providers.openai_compat:TogetherProvider",
    "gguf": "beni.core.morphotactic.distil.providers.gguf:GGUFProvider",
}


def _load_class(spec: str):
    module_name, cls_name = spec.split(":")
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def _init_kwargs(provider_class: Type, kw: Dict[str, Any]) -> Dict[str, Any]:
    """Keep constructor kwargs the selected provider actually accepts.

    Distiller forwards a shared bag (``base_url``, ``gguf_path``, ``n_ctx``, …).
    Providers that do not declare those parameters must not receive them.
    """
    try:
        params = inspect.signature(provider_class.__init__).parameters
    except (TypeError, ValueError):
        return dict(kw)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kw)
    allowed = {
        name
        for name, param in params.items()
        if name not in {"self", "api_key", "model"}
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kw.items() if key in allowed}


def create_provider(name: str, api_key: str = None, model: str = None, **kw) -> base.BaseProvider:
    """Create a Distiller provider from the registry.

    Parameters
    ----------
    name : str
        ``google`` / ``gemini`` / ``openai`` / ``groq`` / ``together`` / ``gguf``.
    api_key : str, optional
        Falls back to the provider's environment variable.
    model : str, optional
        Provider model id.
    """
    spec = PROVIDER_REGISTRY.get(str(name or "").lower())
    if not spec:
        raise ValueError(f"Unsupported provider: {name}. Known: {sorted(PROVIDER_REGISTRY)}")
    provider_class = _load_class(spec)
    return provider_class(api_key, model, **_init_kwargs(provider_class, kw))
