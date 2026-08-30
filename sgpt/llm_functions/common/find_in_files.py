import os, subprocess, shlex
from pydantic import Field, BaseModel
from typing import Any, Dict


class Function(BaseModel):
    """Recursively search files for PATTERN and return matching lines with line numbers."""

    pattern: str = Field(..., description="Regex or fixed string to search for")
    directory: str = Field(
        default=".", description="Directory to recursively search in"
    )
    max_hits: int | None = Field(
        default=None,
        ge=1,
        description="Return at most this many matching lines (across all files)",
    )
    exclude_dirs: list[str] | None = Field(
        default=None,
        description="Directories to skip (passed to grep as --exclude-dir). "
        "Accepts absolute paths or globs, e.g. ['.git','node_modules']",
    )
    file_extensions: list[str] | None = Field(
        default=None,
        description="Only search files whose names match these extensions "
        "(e.g. ['py','proto']). Uses grep --include.",
    )

    @classmethod
    def execute(
        cls,
        *,
        pattern: str,
        directory: str = ".",
        max_hits: int | None = None,
        exclude_dirs: list[str] | None = None,
        file_extensions: list[str] | None = None,
    ) -> str:
        directory = os.path.expandvars(os.path.expanduser(directory))

        base_cmd: list[str] = ["grep", "-R", "-n", "--color=never"]

        # append exclusions
        if exclude_dirs:
            for d in exclude_dirs:
                base_cmd.append(f"--exclude-dir={d}")

        # include only selected extensions
        if file_extensions:
            for ext in file_extensions:
                ext = ext.lstrip("*.")  # clean up
                base_cmd.append(f"--include=*.{ext}")

        # add pattern + target dir
        base_cmd.extend([pattern, directory])

        if max_hits:
            cmd = (
                " ".join(shlex.quote(p) for p in base_cmd)
                + f" | head -n {int(max_hits)}"
            )
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        else:
            proc = subprocess.Popen(
                base_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

        stdout, _ = proc.communicate()
        return f"Exit code: {proc.returncode}, Output: {stdout}"



    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        """Generate OpenAI function schema from Pydantic model."""
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": "find_in_files",
                "description": cls.__doc__.strip() if cls.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }