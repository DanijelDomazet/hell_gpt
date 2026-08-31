#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/home/dd/GDrive/hellgpt/data/chat_cache"
DST_DIR="/home/dd/GDrive/hellgpt/data/chat_cache_backup"

# Timestamp like 2026-07-14_082233
TS="$(date +'%F_%H%M%S')"
ARCHIVE_NAME="chat_cache_${TS}.zip"
TMP_ZIP="${DST_DIR}/.${ARCHIVE_NAME}.tmp"
FINAL_ZIP="${DST_DIR}/${ARCHIVE_NAME}"

# ---- sanity checks ----
if [[ ! -d "$SRC_DIR" ]]; then
  echo "ERROR: source dir not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DST_DIR"

# Create zip in temp file then move into place (atomic-ish)
# -r: recursive, -q: quiet, -9: max compression
(
  cd "$SRC_DIR"
  zip -rq9 "$TMP_ZIP" .
)

mv -f "$TMP_ZIP" "$FINAL_ZIP"

# Optional retention: delete backups older than 30 days
# find "$DST_DIR" -maxdepth 1 -type f -name 'chat_cache_*.zip' -mtime +30 -delete

echo "Backup created: $FINAL_ZIP"
