#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Re-attributes a range of commits from the given starting commit hash to HEAD,
# resetting the author and committer to the current git identity. Signing is
# controlled by the git config (commit.gpgsign). Intended to be run outside the
# container after a Claude Code session.
#
# Usage: ResignCommits.sh <first-commit-hash>

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <first-commit-hash>"
    echo ""
    echo "Re-attributes all commits from <first-commit-hash> to HEAD"
    echo "using the current git user.name and user.email."
    exit 1
fi

FIRST_COMMIT="$1"

# Verify the commit exists
if ! git cat-file -e "${FIRST_COMMIT}^{commit}" 2>/dev/null; then
    echo "Error: '${FIRST_COMMIT}' is not a valid commit in this repository."
    exit 2
fi

# Abort if any merge commits exist in the range
MERGE_COMMITS=$(git rev-list --merges "${FIRST_COMMIT}^..HEAD")
if [ -n "${MERGE_COMMITS}" ]; then
    echo "Error: Merge commits found in the range ${FIRST_COMMIT}..HEAD."
    echo "ResignCommits.sh is not designed to handle merge commits."
    echo ""
    echo "Merge commits in range:"
    git log --oneline --merges "${FIRST_COMMIT}^..HEAD"
    exit 3
fi

echo "Re-attributing commits from ${FIRST_COMMIT} to HEAD..."
AMEND_CMD='git commit --amend --no-edit --reset-author'
git rebase --exec "${AMEND_CMD}" "${FIRST_COMMIT}^"

echo ""
echo "Done. Verify with: git log --show-signature"
