"""SGPT LLM function: gmail_search

Search Gmail using the official Gmail API (OAuth).

Notes:
- Uses OAuth Desktop App client secret JSON.
- Stores/refreshes token locally.
- Returns a compact JSON payload suitable for LLM consumption.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from instructor import OpenAISchema
from pydantic import Field


def _extract_header(headers: List[Dict[str, str]], name: str) -> str:
    name_l = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_l:
            return h.get("value", "")
    return ""


def _get_default_paths() -> tuple[Path, Path]:
    """Resolve secret + token paths from a single required env var.

    Required:
      GMAIL_CREDENTIALS_DIR=/path/to/credentials

    Expected files in that directory:
      gmail_client_secret.json
      gmail_token.json
    """

    cred_dir_raw = os.environ.get("GMAIL_CREDENTIALS_DIR")
    if not cred_dir_raw:
        raise RuntimeError(
            "Missing env var GMAIL_CREDENTIALS_DIR. "
            "Example: export GMAIL_CREDENTIALS_DIR=~/path/to/credentials"
        )
    cred_dir = Path(cred_dir_raw).expanduser()

    secret = cred_dir / "gmail_client_secret.json"
    token = cred_dir / "gmail_token.json"
    return secret, token


def _gmail_service():
    # Lazy imports so SGPT still works if deps are missing.
    from google.auth.transport.requests import Request  # type: ignore
    from google.oauth2.credentials import Credentials  # type: ignore
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
    from googleapiclient.discovery import build  # type: ignore

    client_secret_file, token_file = _get_default_paths()
    if not client_secret_file.exists():
        raise FileNotFoundError(
            f"Gmail client secret not found: {client_secret_file}. "
            "Set SGPT_GMAIL_CLIENT_SECRET_FILE or place the JSON there."
        )

    scope = os.environ.get("GMAIL_SCOPE")
    if not scope:
        raise RuntimeError(
            "Missing env var GMAIL_SCOPE. "
            "Example: export GMAIL_SCOPE=https://www.googleapis.com/auth/gmail.readonly"
        )
    scopes = [scope]

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # This will open a browser on first run.
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secret_file), scopes
            )
            creds = flow.run_local_server(port=0)

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _safe_dt(value: str) -> str:
    # Keep header date as-is, but try to normalize ISO if it looks like an epoch.
    try:
        # Gmail internalDate is epoch millis (string)
        if value.isdigit():
            return datetime.utcfromtimestamp(int(value) / 1000).isoformat() + "Z"
    except Exception:
        pass
    return value


class Function(OpenAISchema):
    """Search Gmail messages via Gmail API (OAuth)."""

    q: str = Field(..., description="Gmail search query")
    max_results: int = Field(5, ge=1, le=200, description="Max number of results (page size)")
    page_token: str | None = Field(
        None,
        description="Optional page token from a previous gmail_search response to fetch the next page.",
    )

    class Config:
        title = "gmail_search"

    @classmethod
    def execute(
        cls, q: str, max_results: int = 5, page_token: str | None = None
    ) -> str:
        service = _gmail_service()

        # 1) list message IDs
        list_req = (
            service.users()
            .messages()
            .list(userId="me", q=q, maxResults=max(1, min(int(max_results), 200)))
        )
        if page_token:
            list_req = list_req.pageToken(page_token)

        resp = list_req.execute()
        msgs = resp.get("messages", []) or []
        next_page_token = resp.get("nextPageToken")

        results: List[Dict[str, Any]] = []
        for m in msgs:
            mid = m.get("id")
            if not mid:
                continue

            full = (
                service.users()
                .messages()
                .get(userId="me", id=mid, format="metadata", metadataHeaders=["From", "To", "Subject", "Date"])
                .execute()
            )
            payload = full.get("payload", {}) or {}
            headers = payload.get("headers", []) or []

            results.append(
                {
                    "id": full.get("id"),
                    "threadId": full.get("threadId"),
                    "from": _extract_header(headers, "From"),
                    "to": _extract_header(headers, "To"),
                    "subject": _extract_header(headers, "Subject"),
                    "date": _extract_header(headers, "Date"),
                    "internalDate": _safe_dt(str(full.get("internalDate", ""))),
                    "snippet": full.get("snippet", ""),
                }
            )

        out = {
            "query": q,
            "result_count": len(results),
            "nextPageToken": next_page_token,
            "results": results,
        }
        return json.dumps(out, ensure_ascii=False, indent=2)
