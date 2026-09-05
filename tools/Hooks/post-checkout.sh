#!/bin/sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Symlink git-ignored personal files from the main worktree into new worktrees

# Only on branch checkouts, e.g. `git worktree add`
[ "$3" = "0" ] && exit 0

main=$(git worktree list --porcelain | sed -n 's/^worktree //p' | head -n 1)
top=$(git rev-parse --show-toplevel) || exit 0
[ "$top" = "$main" ] && exit 0

for file in CMakeUserPresets.json AGENTS.local.md CLAUDE.local.md \
            .claude/settings.local.json; do
    if [ -e "$main/$file" ] && [ ! -e "$top/$file" ]; then
        ln -s "$main/$file" "$top/$file"
    fi
done
