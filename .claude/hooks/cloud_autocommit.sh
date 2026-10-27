#!/usr/bin/env bash
# Stop hook: in Claude Code cloud sessions (claude.ai web or app), commit and push
# after every turn. An idle cloud VM is reclaimed and uncommitted work is lost.
# On the laptop it exits at once; scripts/weekly_commit.sh commits there.
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
set -uo pipefail
cd "$CLAUDE_PROJECT_DIR" || exit 1

# Leave a merge, rebase or cherry-pick that Claude is in the middle of alone.
for f in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply; do
  [ -e "$(git rev-parse --git-path "$f")" ] && exit 0
done

git add -A
if ! git diff --cached --quiet; then
  # Same safety checks as scripts/weekly_commit.sh.
  KEY_RE="(api[_-]?key|token|secret)[^=:]*[=:][[:space:]]*[\"']?[A-Za-z0-9]{20,}"
  if grep -Eiq "$KEY_RE" < <(git diff --cached -U0); then
    git reset -q
    echo "Auto-commit skipped: staged changes look like they contain a key or token." >&2
    exit 1
  fi
  mapfile -t FILES < <(git diff --cached --name-only)
  for f in "${FILES[@]}"; do
    if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 52428800 ]; then
      git reset -q
      echo "Auto-commit skipped: $f is over 50 MB." >&2
      exit 1
    fi
  done
  git commit -q -m "Cloud session snapshot $(TZ=Asia/Singapore date +%Y-%m-%d): ${#FILES[@]} file(s) changed" \
    -m "$(printf '%s\n' "${FILES[@]:0:20}")" || exit 1
fi

# Push anything not on GitHub yet, including commits Claude made itself. Never push to
# main: off a claude/ branch, push to claude/autosave-<session> for the weekly merge.
if [ -n "$(git rev-list -1 HEAD --not --remotes=origin)" ]; then
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
  SID=${CLAUDE_CODE_REMOTE_SESSION_ID:-session}
  case "$BRANCH" in
    claude/*) git push -q -u origin HEAD ;;
    *) git push -q origin "HEAD:refs/heads/claude/autosave-${SID##*_}" ;;
  esac || { echo "Auto-commit: push failed; the commit exists only in this session." >&2; exit 1; }
fi
