#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

run_dir=`pwd`

cd @CMAKE_SOURCE_DIR@
git_commit_hash=`git describe --abbrev=0 --always --tags`
git_branch=`git rev-parse --abbrev-ref HEAD`

cd $run_dir

"$@" -DGIT_COMMIT_HASH=$git_commit_hash -DGIT_BRANCH=$git_branch
