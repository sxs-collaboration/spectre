#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Claude Code PreToolUse hook: capture repo baseline before first modification.
#
# Runs on Edit/Write/Bash. On first invocation, records HEAD commit hash and
# md5 hashes of the working tree state for relevant source files (.?pp, .py,
# CMakeLists.txt), excluding .claude/, .skills/, and .codex/ directories.
# Subsequent invocations exit immediately.
# The snapshot is used by StopVerify.sh to detect whether Claude changed files.
#
# Shared environment and helpers (session id, snapshot path, cleanup_stale_files,
# compute_state) live in Common.sh.

set -euo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.claude/hooks/Common.sh
source "${HOOK_DIR}/Common.sh"

# Clean up stale snapshots from any session for this project.
cleanup_stale_files

# Fast path: snapshot already exists (already captured for this session)
if [[ -f "$SNAPSHOT" ]]; then
  exit 0
fi

# Capture baseline (runs once per session).
compute_state > "$SNAPSHOT"

exit 0
