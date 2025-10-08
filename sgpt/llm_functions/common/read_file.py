import pathlib, os, itertools
from typing import Optional

from instructor import OpenAISchema
from pydantic import Field

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SAFE_ROOT = pathlib.Path("/home/dd").resolve()


class Function(OpenAISchema):
    """Reads text from a file and returns it as UTF-8.

    Parameters `start` and `end` are **1-based line numbers**:
      • `start` is inclusive (defaults to 1 – the first line)
      • `end`   is inclusive as well; pass `None` to read to EOF.
    """

    file_path: str = Field(..., description="Absolute or ~/relative path to the file")
    start: Optional[int] = Field(
        1,
        ge=1,
        description="Line number to start reading from (1-based, inclusive)",
    )
    end: Optional[int] = Field(
        None,
        ge=1,
        description="Line number to stop reading at (inclusive, 1-based)",
    )

    class Config:
        title = "read_file"

    # ------------------------------------------------------------------
    # Executor
    # ------------------------------------------------------------------

    @classmethod
    def execute(
        cls,
        file_path: str,
        start: int = 1,
        end: Optional[int] = None,
        _max_chars: int = 10_000,
    ) -> str:
        """Return text from *start* to *end* lines (inclusive)."""

        try:
            # Resolve path and enforce jail
            p = pathlib.Path(file_path).expanduser().resolve()
            if SAFE_ROOT not in p.parents and p != SAFE_ROOT:
                return (
                    f"Error: '{file_path}' is outside the permitted root "
                    f"({SAFE_ROOT})."
                )

            if not p.exists():
                return f"Error: file '{file_path}' does not exist."
            if not p.is_file():
                return f"Error: path '{file_path}' is not a regular file."

            # Read with best-effort UTF-8 decoding
            with p.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()

            # Convert 1-based → Python slice indexes
            start_idx = max(start - 1, 0)
            end_idx = (
                end if end is not None else None
            )  # NOTE: inclusive in API → exclusive in slice

            slice_lines = lines[start_idx:end_idx]
            text = "".join(slice_lines)

            # Cap very long output
            if len(text) > _max_chars:
                text = (
                    text[:_max_chars]
                    + f"\n...[truncated; total {len(text)} chars in slice]..."
                )

            return text or "(file empty in requested range)"

        except Exception as e:
            return f"Error reading '{file_path}': {e}"
