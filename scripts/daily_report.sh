#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT/config/betterbot.yaml}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
BETTERBOT_CONFIG="$CONFIG_PATH" python3 -c "from betterbot.utils.daily_report import send_daily_report; send_daily_report()"
