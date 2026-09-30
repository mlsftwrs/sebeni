import time
import json
import logging
from rich.console import Console
from pydantic import ValidationError
from google import genai
from typing import Dict, Union, Any
from beni.utils import config as cfg
from beni.core.morphotactic.distil.providers import base

logger = logging.getLogger(__name__)

class GoogleProvider(base.BaseProvider):
    """Google Gemini provider."""

    def __init__(
            self, api_key=None, 
            model="gemma-4-26b-a4b-it", language: str= None, 
            temperature: float =cfg.TEMPERATURE, cache_contents: Union[None, Dict]=None, 
            TTL=cfg.CACHE_TTL, vertex=False, project_id: str=None, region: str=None):

        super().__init__(api_key or cfg.GOOGLE_API, model)
        self.project_id = project_id or cfg.google_project_id()
        self.region = region or cfg.google_location()
        self.vertex = bool(vertex)
        
        self.temperature = temperature
        self.language = language
        self.cache_name = f"{self.language}_{cfg.CACHE_NAME}"
        self.ttl = TTL
        self._client = self.initialize()
        self.cache = self.__get_cache(contents=cache_contents)
        self._cache_created_at = 0


    def initialize(self):
        """ Initialize Google Gemini client. """
        if not self.vertex and not self.api_key:
            raise ValueError(
                "Google authentication missing. Set GOOGLE_API_KEY, configure ADC "
                "with vertex: true, or use distillation.backend: algorithmic."
            )
        if self.vertex and not self.project_id:
            raise ValueError(
                "Vertex/ADC requires GOOGLE_CLOUD_PROJECT or GOOGLE_PROJECT_ID "
                "(or ADC quota_project_id from "
                "`gcloud auth application-default set-quota-project`)."
            )
        if self.vertex:
            logger.info(
                "Google Distiller: Vertex AI (ADC) project=%s location=%s model=%s",
                self.project_id,
                self.region,
                self.model,
            )
            return self.__vertex_client()
        logger.info("Google Distiller: Gemini Developer API (AI Studio) model=%s", self.model)
        return self.__api_client()


    def set_vertex(self, vertex: bool):
        """ Set Vertex AI mode. """
        self.vertex = vertex
        self._client = self.initialize()

    def set_cache_contents(self, langmeta, gram_path, dict_path):
        """ Set cache contents. """
        from beni.core.morphotactic.distil import DistilSysPrompt
        contents = DistilSysPrompt.get_cache_contents(langmeta, gram_path, dict_path)

        self.cache = self.__get_cache(contents=contents)

    def __api_client(self):
        """ Initialize Google Gemini API client. """
        return genai.Client(api_key=self.api_key)

    def __vertex_client(self):
        """ Initialize Google Gemini Vertex AI client. """
        return genai.Client(vertexai=True, project=self.project_id, location=self.region)


    @property
    def capability(self) -> base.ProviderCapability:
        return base.ProviderCapability.BOTH
    
    def __get_cache(self, cache_name: str=None, model=None, contents=None):
        """ Get live cache if exists else lazy load contexts to context cache """

        if contents is None: 
            print(f"Cache contents not provided for {self.cache_name}, skipping cache creation")
            return None

        # Google Gemini explicit caching requires a minimum token threshold of 1024 tokens (~3500-4000 characters).
        # Skip cache creation defensively if contents are too small to avoid 400 INVALID_ARGUMENT.
        if isinstance(contents, str) and len(contents) < 3500:
            logger.info(
                "Cache contents size (%d characters) is below explicit caching threshold (~1024 tokens); skipping cache creation.",
                len(contents),
            )
            return None

        cache_name = self.cache_name if not cache_name else cache_name
        model = model if model else self.model

        try:
            for cache in self._client.caches.list():
                if cache.display_name == cache_name:
                    self._client.caches.update(
                        name=cache.name,
                        config={"ttl": "3600s"} 
                    )
                    self.cache = cache
                    print(f"Cache found for {cache_name} --> {cache}")
                    return cache

            print(f"Cache not found for {cache_name}, creating new cache")
            cache = self._client.caches.create(
                model=model,
                config=genai.types.CreateCachedContentConfig(contents=[contents], ttl=self.ttl,
                    display_name=cache_name))
        except Exception as exc:
            logger.warning(
                "Google context cache unavailable (%s); continuing without cache.",
                exc,
            )
            self.cache = None
            return None

        self.cache = cache
        return self.cache

    def create_cache(self, contents: Union[None, str]):
        return self.__get_cache(cache_name=self.cache_name, model=self.model, contents=contents)

    @property
    def valid_cache(self):
        now = time.time()
        # Recreate if missing or within 60 seconds of expiring
        if not self.cache or (now - self._cache_created_at) > (self.ttl - 60):
            self._cache = self.create_cache(ttl=f"{self._ttl_seconds}s")
            self._cache_created_at = now
        return self._cache

    def clear_caches(self):
        """ Clear all caches for the current model. """

        for cache in self._client.caches.list():
                name = cache.name
                print(f"Cache deleting for {cache.display_name} --> {cache.name}")
                self._client.caches.delete(name=name)

    def generate(self, prompt: str=None, sys_instruct: str=None, indicator: bool=False) -> Dict[str, Any]:
        from beni.core.morphotactic.distil import DistilOutput
        """ Generate JSON response from Google Gemini. """

        data: DistilOutput = None
        config_kwargs = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "response_schema": DistilOutput.model_json_schema(),
        }
        if self.cache is not None:
            config_kwargs["cached_content"] = self.cache.name
        config = genai.types.GenerateContentConfig(**config_kwargs)

        if(indicator):
            console = Console()

            with console.status(f"[bold blue]Generating distillation response from model:{self.model}...", spinner="dots"):
                response = self._client.models.generate_content(
                    model=self.model, 
                    contents=prompt, 
                    config=config
                )
        else:
            response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        
        if(isinstance(response, str)):
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON response: {response}")
                data = {}
        else:
            data = response.parsed if hasattr(response, 'json') else {}

        data = data.parsed if hasattr(data, 'parsed') else data

        try:
            result  = DistilOutput.model_validate(data)
        except ValidationError as e:
            print(f"Validation error: {e}")
            return {}
        return result
