#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

. @CMAKE_SOURCE_DIR@/tools/FileTestDefs.sh

###############################################################################
# Get list of non-deleted file names, which we need below.
commit_files=()
while IFS= read -r -d '' file ; do
    if [ -f "${file}" ]; then
        commit_files+=("${file}")
    fi
done < <(@GIT_EXECUTABLE@ diff --cached --name-only -z)

# If no files were added or modified we do not run any of the hooks. If we did
# then rewording commits, and making empty commits would result in the hooks
# searching the entire repository, which is not what we want in general.
if [ ${#commit_files[@]} -eq 0 ]; then
    exit 0
fi

found_error=0

###############################################################################
# Check the file size
@PYTHON_EXECUTABLE@ @CMAKE_SOURCE_DIR@/.git/hooks/CheckFileSize.py
[ "$?" -ne 0 ] && found_error=1

printf '%s\0' "${commit_files[@]}" | run_checks "${standard_checks[@]}"
[ $? -ne 0 ] && found_error=1

###############################################################################
# Use git-clang-format to check for any suspicious formatting of code.
@PYTHON_EXECUTABLE@ @CMAKE_SOURCE_DIR@/.git/hooks/ClangFormat.py

if [ "$found_error" -eq "1" ]; then
    exit 1
fi
