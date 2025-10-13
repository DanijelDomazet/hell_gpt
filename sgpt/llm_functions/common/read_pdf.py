"""
read_pdf.py – Shell-GPT custom function

Extracts plain text from a local PDF file using `pypdf`.

Requirements:
    pip install pypdf
"""

from __future__ import annotations

import pathlib
from typing import Optional

from instructor import OpenAISchema
from pydantic import Field

# ---------------------------------------------------------------------------
# Config – keep the “jail” consistent with read_file.py
# ---------------------------------------------------------------------------

SAFE_ROOT = pathlib.Path("/home/dd").resolve()


# ---------------------------------------------------------------------------
# OpenAI function schema
# ---------------------------------------------------------------------------


class Function(OpenAISchema):
    """Extract text from a PDF between *page_start* and *page_end* (1-based, inclusive)."""

    pdf_path: str = Field(
        ...,
        description="Absolute or ~/relative path to the PDF file",
    )
    page_start: Optional[int] = Field(
        1,
        ge=1,
        description="First page number to extract (1-based)",
    )
    page_end: Optional[int] = Field(
        None,
        ge=1,
        description="Last page number to extract (inclusive)",
    )

    class Config:
        title = "read_pdf"

    # ------------------------------------------------------------------
    # Executor
    # ------------------------------------------------------------------

    @classmethod
    def execute(
        cls,
        pdf_path: str,
        page_start: int = 1,
        page_end: Optional[int] = None,
        _max_chars: int = 20_000,
    ) -> str:
        """Return extracted text (capped at *_max_chars*)."""

        try:
            # Resolve and enforce jail
            p = pathlib.Path(pdf_path).expanduser().resolve()
            if SAFE_ROOT not in p.parents and p != SAFE_ROOT:
                return (
                    f"Error: '{pdf_path}' is outside the permitted root "
                    f"({SAFE_ROOT})."
                )

            if not p.exists():
                return f"Error: file '{pdf_path}' does not exist."
            if p.suffix.lower() != ".pdf":
                return f"Error: '{pdf_path}' is not a PDF file."

            try:
                from pypdf import PdfReader  # type: ignore
            except ModuleNotFoundError:
                return "Error: pypdf library not installed. Run `pip install pypdf`."

            reader = PdfReader(str(p))

            # Sanity-clamp page range
            total_pages = len(reader.pages)
            start_idx = max(page_start - 1, 0)
            end_idx = page_end - 1 if page_end is not None else total_pages - 1
            if start_idx > end_idx or start_idx >= total_pages:
                return "Error: Invalid page range."

            # Extract
            pieces: list[str] = []
            for i in range(start_idx, min(end_idx, total_pages - 1) + 1):
                page = reader.pages[i]
                pieces.append(page.extract_text() or "")

            text = "\n".join(pieces)
            if len(text) > _max_chars:
                text = (
                    text[:_max_chars]
                    + f"\n...[truncated; total {len(text)} chars in slice]..."
                )

            return text or "(no extractable text on selected pages)"

        except Exception as e:
            return f"Error reading PDF '{pdf_path}': {e}"
