#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Claude Code PostToolUse hook: auto-format files after Edit/Write.
#
# - C++ (.cpp, .hpp, .tpp):
#     Edit  -> git-clang-format -f (diff-only formatting)
#     Write -> clang-format -i    (whole-file formatting)
# - Python (.py):
#     black -q && isort -q        (single-file formatting)

# Returns the path to the best available git-clang-format binary.
# Prefers the unversioned `git-clang-format`, then falls back to the
# highest-versioned `git-clang-format-N` found in PATH.
find_git_clang_format() {
  if command -v git-clang-format &>/dev/null; then
    echo "git-clang-format"
    return
  fi
  local best_ver=0 best_bin=""
  IFS=: read -ra dirs <<< "$PATH"
  for dir in "${dirs[@]}"; do
    for bin in "$dir"/git-clang-format-*; do
      [[ -x "$bin" ]] || continue
      local ver="${bin##*-}"
      [[ "$ver" =~ ^[0-9]+$ ]] || continue
      (( ver > best_ver )) && { best_ver=$ver; best_bin=$bin; }
    done
  done
  echo "$best_bin"
}

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
      gcf=$(find_git_clang_format)
      if [[ -n "$gcf" ]]; then
        "$gcf" -f -- "$FILE" 2>/dev/null
      else
        clang-format -i "$FILE" 2>/dev/null
      fi
    fi
    ;;
  *.py)
    black -q "$FILE" 2>/dev/null
    isort -q "$FILE" 2>/dev/null
    ;;
esac

exit 0
