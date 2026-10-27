#!/usr/bin/env bash
# Weekly safety-net commit. Commits only real changes, never empty commits.
set -euo pipefail
REPO="${FYP_REPO:-/c/Users/Remus Reyes/Downloads/deciphering-market-connections folder/deciphering-market-connections}"
LOG="${FYP_LOG:-$HOME/fyp_weekly_commit.log}"   # outside the repo, so the log itself is never a "change"

exec >> "$LOG" 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
cd "$REPO"

git add -A
if git diff --cached --quiet; then
  echo "No changes since the last commit. Nothing committed."
  exit 0
fi

# Safety: refuse to commit anything that looks like an API key, or files over 50 MB.
if git diff --cached -U0 | grep -Eiq '(api[_-]?key|token|secret)[^=]*=[[:space:]]*[A-Za-z0-9]{20,}'; then
  git reset -q
  echo "ABORTED: staged changes look like they contain a key or token. Nothing committed."
  exit 1
fi
mapfile -t FILES < <(git diff --cached --name-only)
for f in "${FILES[@]}"; do
  if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 52428800 ]; then
    git reset -q
    echo "ABORTED: $f is over 50 MB. Nothing committed."
    exit 1
  fi
done

N=${#FILES[@]}
LIST=$(printf '%s\n' "${FILES[@]:0:20}")
git commit -q -m "Weekly snapshot $(date +%Y-%m-%d): $N file(s) changed" -m "$LIST"
git push -q
echo "Committed and pushed $N file(s)."
