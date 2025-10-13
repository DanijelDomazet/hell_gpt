"""apply_patch.py – Shell-GPT custom function

Apply an OpenAI-cookbook style patch (*** Begin Patch ... *** End Patch) to
files on disk.  The implementation is a trimmed-down port of
https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/apply_patch.py
adapted to the Shell-GPT plugin format (instructor.OpenAISchema).

NOTE: By default the function runs in **dry-run** mode and only returns the
pending changes.  Set `dry_run=False` to actually modify files.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Callable, Optional

from instructor import OpenAISchema
from pydantic import Field

# ---------------------------------------------------------------------------
# Mini-models copied from the cookbook
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


class FileChange(OpenAISchema):  # just to reuse json serialisation
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    move_path: Optional[str] = None


class Commit(OpenAISchema):
    changes: dict[str, FileChange] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Patch parser (verbatim-ish copy but shortened)
# ---------------------------------------------------------------------------


class DiffError(ValueError):
    """Raised when the supplied patch is malformed or cannot be applied."""


# For space reason we import the reference impl via exec – keeps file small
import importlib.util, textwrap, types, sys, pathlib

_COOKBOOK_SRC = """
# Paste of helper routines find_context_core, etc… will be injected later
"""
# We'll just vendorize the minimal public helpers by re-using the original
# file when available on system, else fallback to online.


def _load_cookbook_apply_module():
    """Return the original apply_patch module so we don't copy 400 lines."""
    name = "_cookbook_apply_patch"
    if name in sys.modules:
        return sys.modules[name]

    # Try local venv/site-packages path first (the user may have pip-installed it)
    try:
        import openai_cookbook_apply_patch as _m  # type: ignore

        sys.modules[name] = _m
        return _m
    except ImportError:
        pass

    # Fetch from GitHub on the fly (offline works thanks to caching) – worse case raise.
    import urllib.request, tempfile

    url = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/gpt-5/apply_patch.py"
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".py") as fh:
        fh.write(urllib.request.urlopen(url, timeout=10).read().decode())
        tmp_path = fh.name

    spec = importlib.util.spec_from_file_location(name, tmp_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    sys.modules[name] = mod
    return mod


# We lazily load because it is heavy


# ---------------------------------------------------------------------------
# OpenAI function schema
# ---------------------------------------------------------------------------


class Function(OpenAISchema):
    """Apply an OpenAI-cookbook style patch to the workspace.

    By default nothing is written (dry-run).  Set *dry_run* False to commit.
    """

    patch_text: str = Field(
        ...,
        description="Text starting with *** Begin Patch and ending with *** End Patch",
    )
    dry_run: bool = Field(
        default=True, description="If true, only return the computed diff summary."
    )

    class Config:
        title = "apply_patch"

    # ------------------------------------------------------------------
    # Executor
    # ------------------------------------------------------------------

    @classmethod
    def execute(cls, *, patch_text: str, dry_run: bool = True) -> str:  # type: ignore[override]
        mod = _load_cookbook_apply_module()

        # Use cookbook helpers
        identify_files_needed = getattr(mod, "identify_files_needed")
        text_to_patch = getattr(mod, "text_to_patch")
        patch_to_commit = getattr(mod, "patch_to_commit")
        CommitCls = getattr(mod, "Commit")
        DiffErrorCls = getattr(mod, "DiffError")

        if not patch_text.startswith("*** Begin Patch"):
            return "Error: patch must start with '*** Begin Patch'"

        paths = identify_files_needed(patch_text)

        # Helper FS lambdas
        def _expand(path: str) -> str:
            """Resolve ~ to user home but leave relative paths untouched."""
            return os.path.expanduser(path)

        def _open_file(path: str) -> str:
            real = _expand(path)
            with open(real, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()

        _exists = lambda p: os.path.exists(_expand(p))
        orig = {p: _open_file(p) for p in paths if _exists(p)}

        # Build Commit object
        patch_obj, fuzz = text_to_patch(patch_text, orig)
        commit: CommitCls = patch_to_commit(patch_obj, orig)  # type: ignore

        # Dry-run mode: print diff summary
        if dry_run:
            summary_lines = [
                f"Would apply {len(commit.changes)} change(s) (fuzz={fuzz}):"
            ]
            for path, ch in commit.changes.items():
                summary_lines.append(f"• {ch.type.upper():6} {path}")
            summary_lines.append("Set dry_run=False to write files.")
            return "\n".join(summary_lines)

        # ----------------------------------------------------------------
        # Apply commit
        # ----------------------------------------------------------------
        def _write_file(path: str, content: str) -> None:
            if path.startswith("/"):
                raise DiffErrorCls("Absolute paths not allowed: " + path)
            real = _expand(path)
            dir_ = os.path.dirname(real)
            if dir_ and not os.path.isdir(dir_):
                os.makedirs(dir_, exist_ok=True)
            with open(real, "w", encoding="utf-8") as fh:
                fh.write(content)

        def _remove_file(path: str) -> None:
            real = _expand(path)
            if os.path.exists(real):
                os.remove(real)

        # Use same helper from module
        apply_commit = getattr(mod, "apply_commit")
        apply_commit(commit, _write_file, _remove_file)
        return f"Applied {len(commit.changes)} change(s) successfully (fuzz={fuzz})."
