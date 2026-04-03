#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Claude Code PostToolUse hook: auto-format files after Edit/Write.
#
# - C++ (.cpp, .hpp, .tpp):
#     Edit  -> git clang-format -f (diff-only formatting)
#     Write -> clang-format -i    (whole-file formatting)
# - Python (.py):
#     black -q && isort -q        (single-file formatting)

INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

if [[ -z "$FILE" || ! -f "$FILE" ]]; then
  exit 0
fi

case "$FILE" in
  *.cpp|*.hpp|*.tpp)
    if [[ "$TOOL" == "Write" ]]; then
      clang-format -i "$FILE" 2>/dev/null
    else
      git clang-format -f -- "$FILE" 2>/dev/null
    fi
    ;;
  *.py)
    black -q "$FILE" 2>/dev/null
    isort -q "$FILE" 2>/dev/null
    ;;
esac

exit 0
