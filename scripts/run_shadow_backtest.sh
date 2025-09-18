#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT/config/betterbot.yaml}"
LOOKBACK_DAYS="${2:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [[ -n "$LOOKBACK_DAYS" ]]; then
  export BETTERBOT_LOOKBACK_DAYS="$LOOKBACK_DAYS"
fi
BETTERBOT_CONFIG="$CONFIG_PATH" python3 - <<'PY'
import os
from betterbot.utils.config import load_config
from betterbot.utils.shadow_backtest import run_shadow_backtest
cfg = load_config()
lookback = os.getenv('BETTERBOT_LOOKBACK_DAYS')
if lookback is not None:
    days = int(lookback)
else:
    days = int(cfg.get('shadow_backtest.lookback_days', 7))
metrics = run_shadow_backtest(days)
print("Shadow backtest metrics:", metrics)
PY
