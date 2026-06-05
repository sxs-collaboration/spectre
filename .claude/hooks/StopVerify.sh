#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Claude Code Stop hook: verify changes and prompt for build/test.
#
# Compares current repo state against the baseline snapshot captured by
# SnapshotState.sh. Only considers .?pp, .py, and CMakeLists.txt files;
# ignores .claude/, .skills/, and .codex/ directories. If relevant source
# files changed, injects a verification prompt asking Claude to check the
# plan and run build/test commands. Includes a retry counter (max 3) to
# prevent infinite loops.
#
# Shared environment and helpers (session id, snapshot path, cleanup_stale_files,
# compute_state) live in Common.sh.

set -euo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.claude/hooks/Common.sh
source "${HOOK_DIR}/Common.sh"

RETRY_FILE="/tmp/spectre-stop-retries-${PROJECT_HASH}-${SESSION_ID}"
MAX_RETRIES=3

# Clean up stale snapshot/retry files from any session for this project.
cleanup_stale_files

# --- No snapshot → nothing to verify ---
if [[ ! -f "$SNAPSHOT" ]]; then
  exit 0
fi

# --- Check retry limit ---
RETRIES=0
if [[ -f "$RETRY_FILE" ]]; then
  RETRIES=$(cat "$RETRY_FILE")
fi

if (( RETRIES >= MAX_RETRIES )); then
  rm -f "$SNAPSHOT" "$RETRY_FILE"
  exit 0
fi

# --- Compute current state and compare against the baseline snapshot ---
CURRENT_STATE=$(compute_state)
BASELINE=$(cat "$SNAPSHOT")

if [[ "$CURRENT_STATE" == "$BASELINE" ]]; then
  rm -f "$SNAPSHOT" "$RETRY_FILE"
  exit 0
fi

# --- State changed: inject verification prompt ---
RETRIES=$(( RETRIES + 1 ))
echo "$RETRIES" > "$RETRY_FILE"

# Advance the baseline so the next stop-hook run only triggers if NEW
# changes occur after this prompt.
echo "$CURRENT_STATE" > "$SNAPSHOT"

cat >&2 <<'EOF'
STOP HOOK — Verification Required

Your session modified files in this repository. Before finishing:

1. **Plan check**: Re-read the plan (if one was discussed). Verify every item
   has been addressed. If anything is incomplete, continue working.

2. **Unit tests**: Once you have completed the plan, build and run unit tests:
   ninja -j<N> -C <build_dir> unit-tests
   ctest --test-dir <build_dir> -L unit -j<N> --output-on-failure

3. **Non-unit tests**: Once unit tests pass, build and run remaining tests:
   ninja -j<N> -C <build_dir> test-executables
   ctest --test-dir <build_dir> -LE unit -j<N> --output-on-failure

Replace <build_dir> and <N> with the values you used earlier in this session.
If any step fails, fix the issue. Once everything passes, you may stop.
EOF
exit 2
