"""Look up ISO 639 language metadata (name, script, region, family, variants) via Wikidata."""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from beni.utils.config import get_workdir

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "language-metadata-lookup/1.0 (contact: seben@robotsmali.org)"
CACHE_FILE = None  # resolved via cache_file() against the active workdir


def cache_file() -> Path:
    """Language-metadata cache under the active working directory."""
    return get_workdir().data / "language_metadata.json"

# P218 = ISO 639-1, P219 = ISO 639-2, P220 = ISO 639-3
_ISO_PROPERTY_BY_LENGTH = {2: ["P218"], 3: ["P219", "P220"]}


def _run_sparql(query: str, retries: int = 3, backoff: float = 1.5) -> List[Dict[str, Any]]:
    """Execute a SPARQL query against Wikidata and return the result bindings."""
    headers = {"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}
    last_exc: Optional[Exception] = None

    for attempt in range(retries):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_ENDPOINT, params={"query": query}, headers=headers, timeout=30
            )
            resp.raise_for_status()
            return resp.json()["results"]["bindings"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_exc = exc
            time.sleep(backoff * (attempt + 1))

    raise RuntimeError(f"Wikidata SPARQL query failed after {retries} attempts") from last_exc


def _find_entity_id(iso_code: str) -> Optional[str]:
    """Resolve an ISO 639 code to a Wikidata entity id."""
    code = iso_code.strip().lower()
    props = _ISO_PROPERTY_BY_LENGTH.get(len(code))
    if not props:
        raise ValueError("iso_code must be a 2-letter (639-1) or 3-letter (639-2/639-3) code")

    values_clause = " ".join(f"wdt:{p}" for p in props)
    query = f"""
    SELECT ?lang WHERE {{
      VALUES ?prop {{ {values_clause} }}
      ?lang ?prop "{code}" .
    }} LIMIT 1
    """
    rows = _run_sparql(query)
    if not rows:
        return None
    return rows[0]["lang"]["value"].rsplit("/", 1)[-1]


def _core_metadata(entity_id: str) -> Dict[str, Any]:
    """Fetch name, codes, script, region, family parent, and speaker count."""
    query = f"""
    SELECT ?langLabel ?nativeLabel ?iso1 ?iso2 ?iso3
           ?scriptLabel ?countryLabel ?familyLabel ?speakers ?article WHERE {{
      BIND(wd:{entity_id} AS ?lang)
      OPTIONAL {{ ?lang wdt:P218 ?iso1 . }}
      OPTIONAL {{ ?lang wdt:P219 ?iso2 . }}
      OPTIONAL {{ ?lang wdt:P220 ?iso3 . }}
      OPTIONAL {{ ?lang wdt:P1705 ?nativeLabel . }}
      OPTIONAL {{ ?lang wdt:P282 ?script . }}
      OPTIONAL {{ ?lang wdt:P17 ?country . }}
      OPTIONAL {{ ?lang wdt:P495 ?country . }}
      OPTIONAL {{ ?lang wdt:P279 ?family . }}
      OPTIONAL {{ ?lang wdt:P361 ?family . }}
      OPTIONAL {{ ?lang wdt:P1098 ?speakers . }}
      OPTIONAL {{
        ?article schema:about ?lang ;
                 schema:isPartOf <https://en.wikipedia.org/> .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    rows = _run_sparql(query)

    def collect(field: str) -> List[str]:
        seen, out = set(), []
        for r in rows:
            if field in r:
                v = r[field]["value"]
                if v not in seen:
                    seen.add(v)
                    out.append(v)
        return out

    def first(field: str) -> Optional[str]:
        vals = collect(field)
        return vals[0] if vals else None

    return {
        "wikidata_id": entity_id,
        "wikidata_url": f"https://www.wikidata.org/wiki/{entity_id}",
        "wikipedia_url": first("article"),
        "name": first("langLabel"),
        "native_name": first("nativeLabel"),
        "iso_639_1": first("iso1"),
        "iso_639_2": first("iso2"),
        "iso_639_3": first("iso3"),
        "scripts": collect("scriptLabel"),
        "regions": collect("countryLabel"),
        "language_family": collect("familyLabel"),
        "number_of_speakers": first("speakers"),
    }


def _variants(entity_id: str) -> List[str]:
    """Fetch dialects/variants: items marked as subclass-of or part-of this language."""
    query = f"""
    SELECT DISTINCT ?variantLabel WHERE {{
      BIND(wd:{entity_id} AS ?lang)
      {{ ?variant wdt:P279 ?lang . }} UNION {{ ?variant wdt:P361 ?lang . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    rows = _run_sparql(query)
    seen, out = set(), []
    for r in rows:
        v = r["variantLabel"]["value"]
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _load_cache() -> Dict[str, Any]:
    """Load the whole cache file, or an empty dict if it doesn't exist yet."""
    path = cache_file()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Overwrite the cache file with the given dict."""
    path = cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def get_language_metadata(
    iso_code: str, use_cache: bool = True, refresh: bool = False
) -> Dict[str, Any]:
    """Retrieve metadata for a language given an ISO 639-1, 639-2, or 639-3 code, caching results in CACHE_FILE."""
    key = iso_code.strip().lower()
    cache = _load_cache() if use_cache else {}

    if use_cache and not refresh and key in cache:
        logger.info(f"Returning cached metadata for {iso_code}")
        return cache[key]

    entity_id = _find_entity_id(iso_code)
    if entity_id is None:
        logger.error(f"No language found for ISO 639 code {iso_code!r}")
        raise LookupError(f"No language found for ISO 639 code {iso_code!r}")

    data = _core_metadata(entity_id)
    data["variants"] = _variants(entity_id)
    data["queried_code"] = iso_code

    if use_cache:
        cache[key] = data
        _save_cache(cache)

    return data
