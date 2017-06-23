#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

run_dir=`pwd`

cd @CMAKE_SOURCE_DIR@
git_commit_hash=`@GIT_EXECUTABLE@ describe --abbrev=0 --always --tags`
git_branch=`@GIT_EXECUTABLE@ rev-parse --abbrev-ref HEAD`

cd $run_dir

"$@" -DGIT_COMMIT_HASH=$git_commit_hash -DGIT_BRANCH=$git_branch
