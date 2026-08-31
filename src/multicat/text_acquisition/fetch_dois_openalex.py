"""OpenAlex DOI search module (no API key required).

OpenAlex API: https://api.openalex.org/works
Builds a search query using full-text search (title + abstract),
complementing Scopus.
"""
from __future__ import annotations

import time
import requests


_BASE_URL = "https://api.openalex.org/works"
_EMAIL = "mengyujin96@gmail.com"  # polite pool, raises the rate limit


def build_openalex_query(profile_name: str) -> dict:
    """Return the OpenAlex search API parameters (filter + search).

    OpenAlex does not support complex nested boolean expressions, so
    filter=title_and_abstract.search is used for keyword matching, with
    precise filtering left to the Screening Agent.
    """
    # Product keywords (specific to each profile)
    _PRODUCT_QUERIES: dict[str, str] = {
        "lactic_acid": '"lactic acid" OR "lactate" OR "glycolic acid" OR "acetic acid"',
        "levulinic_acid": '"levulinic acid" OR "angelica lactone" OR "GVL"',
        "furfural": "furfural OR furfuraldehyde",
        "c5_sugar": "furfural OR xylose OR hemicellulose",
        "bimetallic": (
            '"lactic acid" OR HMF OR furfural OR "levulinic acid"'
            ' OR "formic acid" OR "acetic acid"'
        ),
        "default": (
            '"lactic acid" OR "5-hydroxymethylfurfural" OR HMF OR furfural'
            ' OR "levulinic acid" OR "formic acid" OR "acetic acid" OR "glycolic acid"'
        ),
    }

    # Substrate keywords (generic)
    substrate = (
        "glucose OR fructose OR xylose OR cellulose OR sucrose OR hemicellulose"
        ' OR "wheat straw" OR bagasse OR "corn stover" OR "rice straw"'
        " OR starch OR mannose OR galactose OR arabinose OR cellobiose"
        ' OR "lignocellulosic biomass" OR "real biomass"'
    )

    # Catalyst keywords (no metal-specific restriction, only the broad heterogeneous-catalysis category)
    catalyst = (
        '"heterogeneous catalyst" OR zeolite OR "solid acid" OR "metal oxide"'
        ' OR "ion exchange resin" OR MOF OR "solid base"'
        ' OR "supported catalyst" OR "carbon catalyst" OR "bifunctional catalyst"'
    )

    product = _PRODUCT_QUERIES.get(profile_name, _PRODUCT_QUERIES["default"])

    # Exclude homogeneous catalysts, fermentation, reviews, and downstream hydrogenation routes
    exclude = (
        'NOT homogeneous NOT fermentation NOT enzymatic NOT review NOT survey'
        ' NOT hydrogenation NOT photocatalysis NOT electrocatalysis'
    )

    search_str = f"({substrate}) AND ({catalyst}) AND ({product}) {exclude}"

    return {
        "search": search_str,
        "filter": "type:article,is_retracted:false",
        "select": "doi,title,primary_location,publication_year",
        "sort": "relevance_score:desc",
        "per-page": 200,
        "mailto": _EMAIL,
    }


def search_openalex(profile_name: str, limit: int = 0) -> list[dict]:
    """Search OpenAlex and return a list of {doi, title, journal, year}.

    Does not pre-filter by publisher -- xml_downloader will attempt the
    download itself, and any source that can supply XML (Elsevier,
    Springer, PMC OA, Wiley OA, etc.) will succeed.
    """
    params = build_openalex_query(profile_name)
    results: list[dict] = []
    cursor = "*"

    print(f"  Searching OpenAlex (profile={profile_name}, limit={limit or 'unlimited'})...")

    while True:
        page_params = dict(params)
        page_params["cursor"] = cursor
        page_params["per-page"] = 200

        resp = requests.get(_BASE_URL, params=page_params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("meta", {})
        works = data.get("results", [])
        total = meta.get("count", 0)

        if not works:
            break

        for w in works:
            doi = w.get("doi")
            if not doi:
                continue
            doi = doi.replace("https://doi.org/", "").strip()
            loc = w.get("primary_location") or {}
            source = loc.get("source") or {}
            results.append({
                "doi": doi,
                "title": w.get("title", ""),
                "journal": source.get("display_name", ""),
                "year": str(w.get("publication_year", "")),
            })

        print(f"  Got {len(results)} results so far (total available: {total})")

        if limit > 0 and len(results) >= limit:
            results = results[:limit]
            break

        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor
        time.sleep(0.2)

    return results
