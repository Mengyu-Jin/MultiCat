"""Semantic Scholar DOI search module (no API key required).

Semantic Scholar API: https://api.semanticscholar.org/graph/v1/paper/search
Free, 1 request per second without a key; an API key raises the rate limit.
"""
from __future__ import annotations

import time
import requests


_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_PROFILE_QUERIES: dict[str, str] = {
    "lactic_acid": (
        "lactic acid heterogeneous catalyst glucose cellulose biomass hydrothermal"
    ),
    "lactic_acid_v2": (
        "lactic acid glycolic acid acetic acid solid catalyst glucose fructose"
    ),
    "levulinic_acid": (
        "levulinic acid heterogeneous catalyst cellulose glucose biomass solid acid"
    ),
    "furfural": (
        "furfural heterogeneous catalyst xylose hemicellulose biomass solid acid"
    ),
    "c5_sugar": (
        "furfural xylose hemicellulose heterogeneous catalyst solid acid zeolite"
    ),
    "bimetallic": (
        "bimetallic catalyst biomass glucose fructose HMF furfural lactic acid levulinic acid"
    ),
    "metal_catalyst": (
        "metal oxide catalyst biomass glucose HMF furfural lactic acid levulinic acid heterogeneous"
    ),
    "default": (
        "heterogeneous catalyst biomass glucose fructose xylose cellulose "
        "HMF furfural lactic acid levulinic acid formic acid acetic acid"
    ),
}


def search_semanticscholar(profile_name: str, limit: int = 0) -> list[dict]:
    """Search Semantic Scholar and return a list of {doi, title, journal, year}.

    Only returns results that have a DOI. No API key required; rate limit is 1 req/s.
    """
    query = _PROFILE_QUERIES.get(profile_name, _PROFILE_QUERIES["default"])
    results: list[dict] = []
    offset = 0
    batch = 100

    print(f"  Searching Semantic Scholar (profile={profile_name}, limit={limit or 'unlimited'})...")

    while True:
        params = {
            "query": query,
            "fields": "externalIds,title,venue,year",
            "offset": offset,
            "limit": batch if limit == 0 else min(batch, limit - len(results)),
        }

        resp = requests.get(_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", [])
        total = data.get("total", 0)

        if not items:
            break

        for item in items:
            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI", "")
            if not doi:
                continue
            doi = doi.strip()
            results.append({
                "doi": doi,
                "title": item.get("title", ""),
                "journal": item.get("venue", ""),
                "year": str(item.get("year", "")),
            })

        print(f"  Got {len(results)} results so far (total available: {total})")

        if limit > 0 and len(results) >= limit:
            results = results[:limit]
            break

        offset += len(items)
        if offset >= total or offset >= 10000:  # S2 returns at most 10000
            break

        time.sleep(1.1)  # 1 req/s without an API key

    return results
