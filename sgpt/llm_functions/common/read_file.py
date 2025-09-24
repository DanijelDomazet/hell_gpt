import pathlib
from typing import Optional

from instructor import OpenAISchema
from pydantic import Field


SAFE_ROOT = pathlib.Path("/home/dd").resolve()  # adjust if you want another jail


class Function(OpenAISchema):
    """
    Reads text from a file and returns it as UTF-8.
    """

    file_path: str = Field(..., description="Absolute or ~/relative path to the file")
    start: Optional[int] = Field(
        0, ge=0, description="Optional byte offset to start reading from"
    )
    end: Optional[int] = Field(
        None, ge=1, description="Optional byte offset to stop reading (exclusive)"
    )

    class Config:
        title = "read_file"

    @classmethod
    def execute(
        cls,
        file_path: str,
        start: int = 0,
        end: Optional[int] = None,
        _max_chars: int = 10_000,
    ) -> str:
        """
        Returns a slice of the file in UTF-8, capped to `_max_chars`
        to avoid flooding the chat.
        """
        p = pathlib.Path(file_path).expanduser().resolve()

        # sandbox: forbid access outside SAFE_ROOT
        if SAFE_ROOT not in p.parents and p != SAFE_ROOT:
            raise ValueError(f"Access to {p} is outside the permitted root {SAFE_ROOT}")

        with p.open("rb") as fh:
            data = fh.read()

        slice_ = data[start:end] if end is not None else data[start:]

        try:
            text = slice_.decode("utf-8")
        except UnicodeDecodeError:
            text = slice_.decode("utf-8", errors="replace")

        # size guard
        if len(text) > _max_chars:
            text = (
                text[:_max_chars]
                + f"\n...[truncated; total {len(slice_)} bytes in slice]..."
            )

        return text
