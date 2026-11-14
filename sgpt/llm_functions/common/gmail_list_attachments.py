"""SGPT LLM function: gmail_list_attachments

List attachments for a Gmail message id.

Env vars (required):
- GMAIL_CREDENTIALS_DIR: directory containing gmail_client_secret.json and gmail_token.json
- GMAIL_SCOPE: must allow reading, e.g. https://www.googleapis.com/auth/gmail.readonly
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field


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


def _walk_parts(part: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    parts = part.get("parts") or []
    for p in parts:
        filename = p.get("filename") or ""
        body = p.get("body") or {}
        attachment_id = body.get("attachmentId")
        if filename and attachment_id:
            out.append(
                {
                    "filename": filename,
                    "mimeType": p.get("mimeType") or "",
                    "attachmentId": attachment_id,
                    "size": body.get("size"),
                    "partId": p.get("partId"),
                }
            )
        # Recurse
        out.extend(_walk_parts(p))
    return out


class Function(BaseModel):
    """List attachments for a given Gmail message id."""

    message_id: str = Field(..., description="Gmail message id")

    @classmethod
    def execute(cls, message_id: str) -> str:
        service = _gmail_service()

        msg = (
            service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )

        payload = msg.get("payload") or {}
        attachments = _walk_parts(payload)
        return json.dumps(
            {
                "message_id": message_id,
                "attachment_count": len(attachments),
                "attachments": attachments,
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        """Generate OpenAI function schema from Pydantic model."""
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": "gmail_list_attachments",
                "description": cls.__doc__.strip() if cls.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }
