"""SGPT LLM function: gmail_download_attachment

Download a Gmail attachment to disk.

Security:
- Only writes inside directory specified by env var GMAIL_DOWNLOAD_DIR.
- Optional overwrite flag.
- Optional max_bytes cap.

Env vars (required):
- GMAIL_CREDENTIALS_DIR: directory containing gmail_client_secret.json and gmail_token.json
- GMAIL_SCOPE: e.g. https://www.googleapis.com/auth/gmail.readonly
- GMAIL_DOWNLOAD_DIR: base directory where attachments are allowed to be saved
"""

from __future__ import annotations

import base64
import os
import json
from pathlib import Path
from typing import Any

from instructor import OpenAISchema
from pydantic import Field


def _get_paths() -> tuple[Path, Path]:
    cred_dir_raw = os.environ.get("GMAIL_CREDENTIALS_DIR")
    if not cred_dir_raw:
        raise RuntimeError(
            "Missing env var GMAIL_CREDENTIALS_DIR (directory with gmail_client_secret.json and gmail_token.json)."
        )
    cred_dir = Path(cred_dir_raw).expanduser()
    return cred_dir / "gmail_client_secret.json", cred_dir / "gmail_token.json"


def _gmail_service():
    from google.auth.transport.requests import Request  # type: ignore
    from google.oauth2.credentials import Credentials  # type: ignore
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
    from googleapiclient.discovery import build  # type: ignore

    client_secret_file, token_file = _get_paths()
    if not client_secret_file.exists():
        raise FileNotFoundError(f"Missing gmail client secret: {client_secret_file}")

    scope = os.environ.get("GMAIL_SCOPE")
    if not scope:
        raise RuntimeError(
            "Missing env var GMAIL_SCOPE (e.g. https://www.googleapis.com/auth/gmail.readonly)."
        )
    scopes = [scope]

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes)
            creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _download_dir() -> Path:
    raw = os.environ.get("GMAIL_DOWNLOAD_DIR")
    if not raw:
        raise RuntimeError(
            "Missing env var GMAIL_DOWNLOAD_DIR (directory where attachments may be saved)."
        )
    p = Path(raw).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_out_path(filename: str) -> Path:
    base = _download_dir()
    # Prevent path traversal by stripping any directories.
    safe_name = Path(filename).name
    out = (base / safe_name).resolve()
    # Ensure resolved path stays within base even with symlinks etc.
    if base not in out.parents and out != base:
        raise RuntimeError("Resolved output path is outside GMAIL_DOWNLOAD_DIR")
    return out


class Function(OpenAISchema):
    """Download a Gmail attachment to GMAIL_DOWNLOAD_DIR."""

    message_id: str = Field(..., description="Gmail message id")
    attachment_id: str = Field(..., description="Attachment id (from gmail_list_attachments)")
    filename: str = Field(
        ..., description="Filename to save as (basename only; directories are stripped)"
    )
    overwrite: bool = Field(False, description="Whether to overwrite an existing file")
    max_bytes: int = Field(
        25_000_000,
        ge=1,
        le=100_000_000,
        description="Maximum allowed download size in bytes",
    )

    class Config:
        title = "gmail_download_attachment"

    @classmethod
    def execute(
        cls,
        message_id: str,
        attachment_id: str,
        filename: str,
        overwrite: bool = False,
        max_bytes: int = 25_000_000,
    ) -> str:
        service = _gmail_service()
        try:
            out_path = _resolve_out_path(filename)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)}, indent=2)

        if out_path.exists() and not overwrite:
            return json.dumps(
                {
                    "ok": False,
                    "error": "File already exists. Set overwrite=true to replace it.",
                    "path": str(out_path),
                },
                indent=2,
            )

        att = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message_id, id=attachment_id)
            .execute()
        )

        data = att.get("data")
        if not data:
            return json.dumps({"ok": False, "error": "No data in attachment."}, indent=2)

        blob = _b64url_decode(data)
        if len(blob) > int(max_bytes):
            return json.dumps(
                {
                    "ok": False,
                    "error": "Attachment exceeds max_bytes.",
                    "size": len(blob),
                    "max_bytes": int(max_bytes),
                },
                indent=2,
            )

        out_path.write_bytes(blob)
        return json.dumps(
            {
                "ok": True,
                "path": str(out_path),
                "size": len(blob),
                "message_id": message_id,
                "attachment_id": attachment_id,
            },
            ensure_ascii=False,
            indent=2,
        )
