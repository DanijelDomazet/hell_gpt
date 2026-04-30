"""
generate_pdf.py – Shell-GPT custom function

Create a PDF from plain text using reportlab (with basic wrapping).

Requirements:
    pip install reportlab
"""

from __future__ import annotations

import pathlib
import textwrap
from typing import Optional

from instructor import OpenAISchema
from pydantic import Field


SAFE_ROOT = pathlib.Path("/home/dd").resolve()


class Function(OpenAISchema):
    """Generate a PDF from plain text and save it."""

    output_pdf_path: str = Field(
        ..., description="Absolute or ~/relative output PDF path"
    )
    text: str = Field(..., description="Plain text to write into the PDF")
    title: Optional[str] = Field(None, description="Optional title")
    wrap_width: int = Field(
        100,
        ge=20,
        le=300,
        description="Wrap line length in characters",
    )

    class Config:
        title = "generate_pdf"

    @classmethod
    def execute(
        cls,
        output_pdf_path: str,
        text: str,
        title: Optional[str] = None,
        wrap_width: int = 100,
        _max_text_chars: int = 200_000,
    ) -> str:
        try:
            if len(text) > _max_text_chars:
                return f"Error: text too large ({len(text)} chars)."

            out = pathlib.Path(output_pdf_path).expanduser().resolve()

            # enforce jail
            if SAFE_ROOT not in out.parents and out != SAFE_ROOT:
                return (
                    f"Error: '{output_pdf_path}' is outside permitted root "
                    f"({SAFE_ROOT})."
                )

            if out.suffix.lower() != ".pdf":
                return "Error: output_pdf_path must end with .pdf"

            out.parent.mkdir(parents=True, exist_ok=True)

            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.pdfgen import canvas
            except ModuleNotFoundError:
                return "Error: reportlab not installed. Run `pip install reportlab`."

            c = canvas.Canvas(str(out), pagesize=A4)
            width, height = A4

            # Layout
            left_margin = 40
            right_margin = 40
            top_margin = 50
            bottom_margin = 50

            body_font = "Helvetica"
            body_size = 11
            heading_font = "Helvetica-Bold"
            heading_size = 12
            title_font = "Helvetica-Bold"
            title_size = 18

            line_height = int(body_size * 1.35)
            heading_line_height = int(heading_size * 1.4)
            title_line_height = int(title_size * 1.2)

            y = height - top_margin

            def new_page() -> None:
                nonlocal y
                c.showPage()
                y = height - top_margin

            def ensure_space(pixels: int) -> None:
                nonlocal y
                if y - pixels < bottom_margin:
                    new_page()

            def draw_line(s: str, font: str, size: int, lh: int) -> None:
                nonlocal y
                ensure_space(lh)
                c.setFont(font, size)
                c.drawString(left_margin, y, s)
                y -= lh

            def wrap_line(raw: str) -> list[str]:
                if raw.strip() == "":
                    return [""]
                return textwrap.wrap(
                    raw,
                    width=wrap_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                ) or [""]

            # Title
            if title:
                for ln in wrap_line(title.strip()):
                    draw_line(ln, title_font, title_size, title_line_height)
                draw_line("", body_font, body_size, line_height)

            # Body with simple heading detection
            prev_blank = True
            for raw in text.splitlines() or [""]:
                is_blank = raw.strip() == ""
                is_heading = (not is_blank) and raw.strip().endswith(":") and len(raw.strip()) <= 60

                if is_heading:
                    # add extra space before headings (unless already at a blank)
                    if not prev_blank:
                        draw_line("", body_font, body_size, line_height)
                    for ln in wrap_line(raw.strip()):
                        draw_line(ln, heading_font, heading_size, heading_line_height)
                else:
                    for ln in wrap_line(raw):
                        draw_line(ln, body_font, body_size, line_height)

                prev_blank = is_blank

            c.save()
            return f"OK: wrote {out}"

        except Exception as e:
            return f"Error generating PDF: {e}"
