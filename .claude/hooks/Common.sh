#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Shared helpers for the Claude Code hook scripts SnapshotState.sh and
# StopVerify.sh. This file is meant to be sourced, not executed: it defines the
# common environment (project hash, session id, snapshot path) and the
# functions both hooks rely on so the logic lives in exactly one place.
#
# The sourcing script is expected to have already run `set -euo pipefail`; the
# functions defined here assume those options are active.

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
PROJECT_HASH=$(echo -n "$PROJECT_DIR" | md5sum | cut -c1-12)

STALE_SECONDS=14400  # 4 hours

# --- Session isolation ---
# Multiple concurrent Claude Code sessions in the same project share /tmp.
# To prevent cross-session interference (one session's baseline being compared
# against another session's changes), we include a session-unique suffix in the
# snapshot filename.
#
# Strategy: walk up the process tree to find the ancestor `claude` process.
# Its PID is unique per Claude Code session and stable across all hook
# invocations within that session. Falls back to PPID if the walk fails
# (e.g., /proc unavailable or process name doesn't match).
find_session_id() {
  local pid=$$
  while [[ "$pid" -gt 1 ]]; do
    local name
    name=$(awk '/^Name:/{print $2}' "/proc/$pid/status" 2>/dev/null) || break
    # Primary: the Claude Code node process sets its process title to "claude"
    # via prctl/process.title, so the /proc Name field is "claude" regardless
    # of command-line flags (--resume, -p, etc.).
    if [[ "$name" == "claude" ]]; then
      echo "$pid"
      return
    fi
    # Secondary: if a future version stops renaming the process, the binary
    # is still node and the first cmdline argument is "claude". Match that.
    if [[ "$name" == "node" ]]; then
      local cmd0
      cmd0=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | head -1) || true
      if [[ "$cmd0" == "claude" || "$cmd0" == */claude ]]; then
        echo "$pid"
        return
      fi
    fi
    pid=$(awk '/^PPid:/{print $2}' "/proc/$pid/status" 2>/dev/null) || break
  done
  # Fallback: use PPID (less reliable across hook types but still better
  # than nothing for avoiding cross-session collisions).
  echo "$PPID"
}

SESSION_ID=$(find_session_id)
SNAPSHOT="/tmp/spectre-snapshot-${PROJECT_HASH}-${SESSION_ID}"

# --- Helper: delete file if older than STALE_SECONDS ---
# Prevents stale snapshots from a crashed or abandoned session from persisting
# indefinitely and causing false comparison results in a new session.
cleanup_stale() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local mtime now
    mtime=$(stat -c %Y "$file" 2>/dev/null || echo 0)
    now=$(date +%s)
    if (( now - mtime > STALE_SECONDS )); then
      rm -f "$file"
    fi
  fi
}

# Clean up stale snapshot and retry files from any session for this project.
# The globs match all session-suffixed files (including the current session's).
cleanup_stale_files() {
  local f
  for f in /tmp/spectre-snapshot-"${PROJECT_HASH}"-* \
           /tmp/spectre-stop-retries-"${PROJECT_HASH}"-*; do
    [[ -e "$f" ]] && cleanup_stale "$f"
  done
  # The trailing [[ -e ]] test above returns non-zero when the glob matches
  # nothing; return success explicitly so this function never trips `set -e`
  # in the caller.
  return 0
}

# Pathspecs: .?pp, .py, CMakeLists.txt files, excluding tool dirs.
# The exclude pathspecs (:!) must come after at least one positive pathspec.
SRC_PATHSPECS=(
  '*.?pp' '*.py' '**/CMakeLists.txt'
  ':!.claude/' ':!.skills/' ':!.codex/'
)

# Print a deterministic hash string describing the repo's current state for the
# relevant source files: HEAD commit plus md5 hashes of the unstaged diff, the
# staged diff, and the untracked source files. SnapshotState.sh records this as
# the baseline; StopVerify.sh recomputes it to detect changes.
#
# Each pipeline stage that uses grep is wrapped in { grep ... || true; }
# because grep returns exit code 1 when zero lines match. Under
# `set -euo pipefail`, an unguarded grep no-match would make the whole
# pipeline return non-zero and terminate the script (leaving a partial
# state string). The || true ensures the pipeline always succeeds while still
# producing the correct md5sum (of empty input when nothing matches).
compute_state() {
  {
    git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo "no-git"
    git -C "$PROJECT_DIR" diff -- "${SRC_PATHSPECS[@]}" 2>/dev/null \
      | md5sum | cut -c1-32
    git -C "$PROJECT_DIR" diff --cached -- "${SRC_PATHSPECS[@]}" 2>/dev/null \
      | md5sum | cut -c1-32
    git -C "$PROJECT_DIR" ls-files --others --exclude-standard 2>/dev/null \
      | { grep -v -E '^(\.claude|\.skills|\.codex)/' || true; } \
      | { grep -E '\..pp$|\.py$|(^|/)CMakeLists\.txt$' || true; } \
      | md5sum | cut -c1-32
  }
}
