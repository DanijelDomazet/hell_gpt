"""read_web_page.py – Shell-GPT custom function (updated w/ graceful 404)

Fetches a remote web page and returns **plain visible text** so the model can
reason over it.

Requirements:
    • beautifulsoup4 – mandatory (ImportError if missing)

Changes:
    • _fetch() no longer raises on HTTP 4xx/5xx; instead returns a friendly
      placeholder that informs the caller the page is inaccessible.
"""

from __future__ import annotations

import re
import ssl
import urllib.request
from typing import Optional

from bs4 import BeautifulSoup
from instructor import OpenAISchema
from pydantic import Field, HttpUrl

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ERROR_TEMPLATE = (
    "[ERROR {code}] Cannot fetch {url}. "
    "Try another page – this one is not accessible."
)


def _fetch(url: str, timeout: int = 10) -> str:
    """Download *url* and return decoded string (best-effort UTF-8).

    On network/HTTP failure returns a short explanatory message instead of
    raising, so the LLM can react gracefully.
    """

    # Prefer requests if available
    try:
        import requests  # type: ignore

        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "shell-gpt-bot"})
        except requests.exceptions.RequestException as exc:  # network errors
            return _ERROR_TEMPLATE.format(code="NET", url=url)

        if resp.status_code >= 400:
            return _ERROR_TEMPLATE.format(code=resp.status_code, url=url)

        raw = resp.content

    except ModuleNotFoundError:
        # Fallback to urllib
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "shell-gpt-bot"})
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as fh:  # type: ignore
                raw = fh.read()
        except urllib.error.HTTPError as err:  # type: ignore
            return _ERROR_TEMPLATE.format(code=err.code, url=url)
        except Exception:
            return _ERROR_TEMPLATE.format(code="NET", url=url)

    # Decode bytes → str
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _html2text(html: str) -> str:
    """Convert HTML → visible text using BeautifulSoup."""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

# ---------------------------------------------------------------------------
# OpenAI function schema
# ---------------------------------------------------------------------------


class Function(OpenAISchema):
    """Downloads a web page and returns its textual content (UTF-8)."""

    url: HttpUrl = Field(..., description="Full HTTP/HTTPS URL of the page to download")
    start: Optional[int] = Field(0, ge=0, description="Character offset to start from")
    end: Optional[int] = Field(None, ge=1, description="Character offset to stop at (exclusive)")

    class Config:
        title = "read_web_page"

    @classmethod
    def execute(
        cls,
        url: str,
        start: int = 0,
        end: Optional[int] = None,
        _max_chars: int = 20_000,
    ) -> str:
        """Fetch URL, convert to text, return slice (capped)."""

        html = _fetch(url)
        # If _fetch already returned an error message, propagate as-is
        if html.startswith("[ERROR"):
            return html

        text = _html2text(html)
        slice_ = text[start:end] if end is not None else text[start:]
        if len(slice_) > _max_chars:
            slice_ = slice_[:_max_chars] + f"\n...[truncated; total {len(text)} chars]..."
        return slice_
