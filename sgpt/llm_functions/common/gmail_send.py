"""Gmail send message tool.

Sends an email via Gmail API using OAuth credentials.
"""

from __future__ import annotations

import base64
import json
import os
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _gmail_service():
    # Lazy imports so hgpt still works if deps are missing unless tool invoked.
    from google.auth.exceptions import RefreshError  # type: ignore
    from google.auth.transport.requests import Request  # type: ignore
    from google.oauth2.credentials import Credentials  # type: ignore
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
    from googleapiclient.discovery import build  # type: ignore

    scopes = [os.getenv("GMAIL_SCOPE", "https://www.googleapis.com/auth/gmail.send")]
    cred_dir = Path(os.getenv("GMAIL_CREDENTIALS_DIR", "~/.config/hell_gpt/credentials")).expanduser()
    token_file = cred_dir / "gmail_token.json"
    client_secret_file = cred_dir / "gmail_client_secret.json"

    if not client_secret_file.exists():
        raise FileNotFoundError(
            f"Missing Gmail client secret: {client_secret_file}. Set GMAIL_CREDENTIALS_DIR."
        )

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes)

    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                raise RefreshError("No valid cached credentials")
        except RefreshError:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes)
            creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


class Function(BaseModel):
    """Send an email using Gmail API."""

    to: List[str] = Field(..., description="Recipient email addresses")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body (plain text)")
    cc: Optional[List[str]] = Field(default=None, description="CC email addresses")
    bcc: Optional[List[str]] = Field(default=None, description="BCC email addresses")
    reply_to: Optional[str] = Field(default=None, description="Reply-To address")

    @classmethod
    def execute(
        cls,
        *,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
    ) -> str:
        service = _gmail_service()

        msg = EmailMessage()
        msg["To"] = ", ".join(to)
        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)
        if reply_to:
            msg["Reply-To"] = reply_to
        msg["Subject"] = subject
        msg.set_content(body)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

        try:
            sent = (
                service.users()
                .messages()
                .send(userId="me", body={"raw": raw})
                .execute()
            )
            return json.dumps({"ok": True, "id": sent.get("id"), "threadId": sent.get("threadId")}, indent=2)
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)}, indent=2)

    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": "gmail_send",
                "description": cls.__doc__.strip() if cls.__doc__ else "",
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }
