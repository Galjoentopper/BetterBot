#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT/config/betterbot.yaml}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
readarray -t CFG < <(BETTERBOT_CONFIG="$CONFIG_PATH" python3 - <<'PY'
from betterbot.utils.config import load_config
cfg = load_config()
flags = cfg.get('backup.rsync_flags', ['-avz'])
for flag in flags:
    print(f"FLAG:{flag}")
print(f"TARGET:{cfg.get('backup.target', '')}")
PY
)
FLAGS=()
TARGET=""
for line in "${CFG[@]}"; do
    if [[ "$line" == FLAG:* ]]; then
        FLAGS+=("${line#FLAG:}")
    elif [[ "$line" == TARGET:* ]]; then
        TARGET="${line#TARGET:}"
    fi
done
[[ ${#FLAGS[@]} -eq 0 ]] && FLAGS=(-avz)
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
SOURCE="$ROOT/models/"
if [[ ! -d "$SOURCE" ]]; then
    echo "Models directory not found: $SOURCE" >&2
    exit 1
fi
if [[ -z "$TARGET" ]]; then
    DEST="$ROOT/backups/$TIMESTAMP/"
    mkdir -p "$DEST"
else
    DEST="$TARGET"
fi
rsync "${FLAGS[@]}" "$SOURCE" "$DEST"
echo "Models backed up to $DEST"
