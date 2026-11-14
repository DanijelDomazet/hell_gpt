#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: backup_chat_cache.sh [--today]

Options:
  --today   Create/overwrite chat_cache_today.zip (instead of timestamped name)
EOF
}

SRC_DIR="/home/dd/GDrive/hellgpt/data/chat_cache"
DST_DIR="/home/dd/GDrive/hellgpt/data/chat_cache_backup"

# Timestamp like 2026-07-14_082233
TS="$(date +'%F_%H%M%S')"

ARCHIVE_NAME="chat_cache_${TS}.zip"

# ---- args ----
MODE="timestamp"
if [[ "${1-}" == "--today" ]]; then
  MODE="today"
  shift
elif [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
elif [[ "${1-}" != "" ]]; then
  echo "ERROR: Unknown argument: $1" >&2
  usage >&2
  exit 2
fi

if [[ "$MODE" == "today" ]]; then
  ARCHIVE_NAME="chat_cache_today.zip"
fi

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
