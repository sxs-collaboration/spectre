#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Claude Code PostToolUse hook: block on compiler warnings in build output.
#
# Runs after every Bash tool call. Scans stdout+stderr for clang/gcc-style
# diagnostic warnings of the form:
#   /path/to/file.cpp:42:10: warning: message [-Wflag]
#
# If any warnings are found, exits 2 to inject a message into Claude's
# context, requiring all warnings to be fixed before the session continues.
# ANSI color codes are stripped before matching.

set -euo pipefail

INPUT=$(cat)

# Only process Bash tool calls
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')
if [[ "$TOOL" != "Bash" ]]; then
  exit 0
fi

# Extract the tool output.
# tool_response may be a plain string, an object with an "output" key,
# or an object with a "content" array of {type,text} blocks.
RESPONSE_TYPE=$(echo "$INPUT" | jq -r '.tool_response | type')
if [[ "$RESPONSE_TYPE" == "string" ]]; then
  OUTPUT=$(echo "$INPUT" | jq -r '.tool_response')
else
  CONTENT_TYPE=$(echo "$INPUT" | jq -r '.tool_response.content | type')
  if [[ "$CONTENT_TYPE" == "array" ]]; then
    OUTPUT=$(echo "$INPUT" | jq -r \
      '[.tool_response.content[] | select(.type == "text") | .text]
       | join("")')
  else
    OUTPUT=$(echo "$INPUT" | jq -r \
      '.tool_response.output // .tool_response.content
       // .tool_response.result // ""')
  fi
fi

# Strip ANSI escape codes (ninja/clang emit colored output by default)
CLEAN=$(printf '%s' "$OUTPUT" | sed 's/\x1b\[[0-9;]*[mK]//g')

# Match clang/gcc diagnostic warnings: file:line:col: warning: ...
# This pattern is specific enough that it won't appear in ctest, git, or
# other tool output, so no command-type detection is needed.
#
# The grep is wrapped in { grep ... || true; } because grep returns exit
# code 1 when zero lines match. Under `set -euo pipefail`, an unguarded
# grep no-match would terminate the script with exit 1 before reaching
# the emptiness check below — making the if/exit 0 path dead code.
WARNINGS=$(printf '%s' "$CLEAN" \
  | { grep -E '[^:]+:[0-9]+:[0-9]+: warning:' || true; } \
  | head -100)

if [[ -z "$WARNINGS" ]]; then
  exit 0
fi

COUNT=$(printf '%s\n' "$WARNINGS" | wc -l)

cat >&2 <<EOF
COMPILER WARNINGS — Fix before continuing

${COUNT} warning(s) found in the build output:

${WARNINGS}

All compiler warnings must be resolved. Fix each warning, rebuild, and
verify the output is warning-free before proceeding.
EOF

exit 2
