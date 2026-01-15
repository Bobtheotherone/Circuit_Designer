#!/usr/bin/env bash
set -euo pipefail

: "${MSG:?Set MSG='your commit message' before running}"
: "${GIT_AUTHOR_NAME:?Set GIT_AUTHOR_NAME}"
: "${GIT_AUTHOR_EMAIL:?Set GIT_AUTHOR_EMAIL}"
: "${GIT_AUTH_TOKEN:?Set GIT_AUTH_TOKEN}"
: "${GIT_REMOTE_URL:?Set GIT_REMOTE_URL}"

# Ensure identity is set (repo-local)
git config user.name  "$GIT_AUTHOR_NAME"
git config user.email "$GIT_AUTHOR_EMAIL"

# Ensure origin uses tokenized remote (HTTPS)
git remote set-url origin "$GIT_REMOTE_URL"

# Stage all changes
git add -A

# Commit (no-op if nothing staged)
if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi

git commit -m "$MSG"

# Push current branch
branch="$(git rev-parse --abbrev-ref HEAD)"
git push origin "$branch"
