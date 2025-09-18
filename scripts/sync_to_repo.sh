#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${1:-origin}"
BRANCH="${2:-main}"
COMMIT_MESSAGE="${3:-Automated BetterBot sync}"
SSH_KEY="${BETTERBOT_DEPLOY_KEY:-$HOME/.ssh/betterbot_deploy_key}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git commit -m "$COMMIT_MESSAGE"
else
    echo "No changes to commit."
fi
GIT_SSH_COMMAND="ssh -i $SSH_KEY -o IdentitiesOnly=yes" git push "$REMOTE" "$BRANCH"
