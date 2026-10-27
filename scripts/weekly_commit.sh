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
else
  # Safety: refuse to commit anything that looks like an API key, or files over 50 MB.
  # grep reads a process substitution, not a pipe: with pipefail, grep -q stopping at
  # the first match would make the pipeline fail and let the key through.
  KEY_RE="(api[_-]?key|token|secret)[^=:]*[=:][[:space:]]*[\"']?[A-Za-z0-9]{20,}"
  if grep -Eiq "$KEY_RE" < <(git diff --cached -U0); then
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
  echo "Committed $N file(s)."
fi

# Replay local commits on top of anything pushed from elsewhere (cloud sessions, and the
# weekly cloud routine that merges them into main), then push everything unpushed,
# including a commit whose push failed in an earlier week.
git fetch -q origin
BEHIND=$(git rev-list --count 'HEAD..@{u}')
if ! git rebase -q '@{u}'; then
  git rebase --abort 2>/dev/null || true
  echo "ABORTED: local commits conflict with origin. Nothing pushed; local commits kept. Fix by hand: git pull --rebase"
  exit 1
fi
[ "$BEHIND" -eq 0 ] || echo "Brought in $BEHIND commit(s) from origin."
AHEAD=$(git rev-list --count '@{u}..HEAD')
if [ "$AHEAD" -gt 0 ]; then
  git push -q
  echo "Pushed $AHEAD commit(s)."
else
  echo "Nothing to push."
fi
