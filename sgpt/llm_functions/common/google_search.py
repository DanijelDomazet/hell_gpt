"""google_search.py – Shell-GPT custom function

Uses SerpApi (https://serpapi.com/search-api) to perform a Google search and
returns a compact list of result links (optionally with titles/snippets).

Why SerpApi?
    Because Google loves blocking bots and SerpApi gets to be the designated
    driver 😎.

Auth:
    Provide API key via SERPAPI_API_KEY env var (recommended), or pass `api_key`.

Dependencies:
    pip install serpapi
"""

from __future__ import annotations

import os
from typing import Optional

import serpapi
from instructor import OpenAISchema
from pydantic import Field


class Function(OpenAISchema):
    """Search Google via SerpApi and return up to `num_results` result links."""

    query: str = Field(..., description="Search query")
    num_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="How many organic result links to return (max 100)",
    )
    location: Optional[str] = Field(
        default=None,
        description="SerpApi 'location' (e.g. 'Austin, Texas, United States')",
    )
    hl: Optional[str] = Field(
        default="en",
        description="Host language (SerpApi 'hl'), e.g. 'en'",
    )
    gl: Optional[str] = Field(
        default=None,
        description="Country (SerpApi 'gl'), e.g. 'us'",
    )
    safe: Optional[str] = Field(
        default=None,
        description="SafeSearch: 'active' or 'off' (SerpApi 'safe')",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="SerpApi API key. Prefer env var SERPAPI_API_KEY.",
    )
    include_metadata: bool = Field(
        default=False,
        description="If true, include title/snippet per result (still returns text).",
    )

    class Config:
        title = "google_search"

    @classmethod
    def execute(
        cls,
        *,
        query: str,
        num_results: int = 10,
        location: str | None = None,
        hl: str | None = "en",
        gl: str | None = None,
        safe: str | None = None,
        api_key: str | None = None,
        include_metadata: bool = False,
    ) -> str:
        key = api_key or os.getenv("SERPAPI_API_KEY")
        if not key:
            return (
                "[ERROR] Missing SerpApi API key. Set SERPAPI_API_KEY env var "
                "or pass api_key."
            )

        params: dict = {
            "engine": "google",
            "q": query,
            "api_key": key,
            # Ask for more internally so we can slice reliably.
            # Google engine supports 'num' up to 100.
            "num": min(int(num_results), 100),
        }
        if location:
            params["location"] = location
        if hl:
            params["hl"] = hl
        if gl:
            params["gl"] = gl
        if safe:
            params["safe"] = safe

        try:
            results = serpapi.search(**params)  # returns SerpResults (dict-like)
        except Exception as exc:
            return f"[ERROR] SerpApi request failed: {type(exc).__name__}: {exc}"

        organic = results.get("organic_results") or []
        if not organic:
            # Provide some context if SerpApi returned an error payload.
            err = results.get("error") or results.get("search_metadata", {}).get("status")
            return f"No organic results found. Details: {err!r}"

        lines: list[str] = []
        for i, item in enumerate(organic[: int(num_results)], start=1):
            link = item.get("link")
            if not link:
                continue

            if include_metadata:
                title = (item.get("title") or "").strip()
                snippet = (item.get("snippet") or "").strip()
                bits = [f"{i}. {link}"]
                if title:
                    bits.append(f"   title: {title}")
                if snippet:
                    bits.append(f"   snippet: {snippet}")
                lines.append("\n".join(bits))
            else:
                lines.append(f"{i}. {link}")

        return "\n".join(lines) if lines else "No links found in organic_results."
