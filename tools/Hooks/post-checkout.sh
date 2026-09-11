#!/bin/sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Symlink git-ignored personal files from the main worktree into new worktrees

# Nothing to do in the main worktree, where .git is a directory
[ -d .git ] && exit 0

# The main worktree is listed first
main=$(git worktree list --porcelain | sed -n 's/^worktree //p' | head -n 1)

for file in CMakeUserPresets.json AGENTS.local.md CLAUDE.local.md \
            .claude/settings.local.json; do
    # -e is false for dangling symlinks, which -f then replaces
    if [ -e "$main/$file" ] && [ -d "$(dirname "$file")" ] \
        && [ ! -e "$file" ]; then
        ln -sf "$main/$file" "$file"
    fi
done
