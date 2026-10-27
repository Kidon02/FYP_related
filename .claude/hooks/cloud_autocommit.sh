#!/usr/bin/env bash
# Stop hook: in Claude Code cloud sessions (claude.ai web or app), commit and push
# after every turn. An idle cloud VM is reclaimed and uncommitted work is lost.
# On the laptop it exits at once; scripts/weekly_commit.sh commits there.
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
set -uo pipefail
cd "$CLAUDE_PROJECT_DIR" || exit 1

# A refusal exits 2, so Claude sees the reason and fixes it before the cloud platform's
# own Stop hook (which asks Claude to commit and push) gets it committed. If Claude is
# already continuing because of a Stop hook, exit 1 (a notice only) so it cannot loop.
INPUT=""
[ -t 0 ] || INPUT=$(cat)
case "$INPUT" in *'"stop_hook_active":true'*|*'"stop_hook_active": true'*) REFUSE=1 ;; *) REFUSE=2 ;; esac
refuse() { git reset -q; echo "Auto-commit refused: $1 Fix this before committing or pushing." >&2; exit "$REFUSE"; }
# The platform's Stop hook runs git at the same moment, so retry while index.lock is held.
retry() { for _ in 1 2 3 4 5; do "$@" && return 0; sleep 1; done; return 1; }

# Leave a merge, rebase or cherry-pick that Claude is in the middle of alone.
for f in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD rebase-merge rebase-apply; do
  [ -e "$(git rev-parse --git-path "$f")" ] && exit 0
done

retry git add -A || exit 1
if ! git diff --cached --quiet; then
  # Same safety checks as scripts/weekly_commit.sh, per file so the refusal names it.
  KEY_RE="(api[_-]?key|token|secret)[^=:]*[=:][[:space:]]*[\"']?[A-Za-z0-9]{20,}"
  mapfile -t FILES < <(git diff --cached --name-only)
  for f in "${FILES[@]}"; do
    if grep -Eiq "$KEY_RE" < <(git diff --cached -U0 -- "$f"); then
      refuse "$f looks like it contains an API key or token, and the repo is public. Read keys from environment variables instead."
    fi
    if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 52428800 ]; then
      refuse "$f is over 50 MB; delete it or add it to .gitignore."
    fi
  done
  retry git commit -q -m "Cloud session snapshot $(TZ=Asia/Singapore date +%Y-%m-%d): ${#FILES[@]} file(s) changed" \
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
