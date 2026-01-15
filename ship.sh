#!/usr/bin/env bash
set -euo pipefail

msg="${*:-}"
if [[ -z "$msg" ]]; then
  echo "Usage: ./ship.sh \"commit message\""
  exit 2
fi

git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi

git commit -m "$msg"
branch="$(git rev-parse --abbrev-ref HEAD)"
git push origin "$branch"
echo "Pushed: $branch"
