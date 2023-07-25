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
@Python_EXECUTABLE@ @CMAKE_SOURCE_DIR@/.git/hooks/CheckFileSize.py
[ "$?" -ne 0 ] && found_error=1

printf '%s\0' "${commit_files[@]}" | \
    run_checks "${standard_checks[@]}" prevent_run_single_test_changes
[ $? -ne 0 ] && found_error=1

###############################################################################
# Use git-clang-format to check for any suspicious formatting of code.
@CLANG_FORMAT_BIN@ --version > /dev/null
if [ $? -eq 0 ]; then
    clang_format_diffstat=$(@GIT_EXECUTABLE@ --no-pager \
        clang-format --binary @CLANG_FORMAT_BIN@ --diffstat --quiet)
    # Clang-format didn't always return the right exit code before version 15,
    # so we check the diffstat output instead (see issue:
    # https://github.com/llvm/llvm-project/issues/54758)
    if [ -n "$clang_format_diffstat" ]; then
        found_error=1
        @GIT_EXECUTABLE@ clang-format --binary @CLANG_FORMAT_BIN@ --diff \
            > @CMAKE_SOURCE_DIR@/.clang_format_diff.patch
        echo "Found C++ formatting errors:"
        echo "$clang_format_diffstat"
        echo "Please run 'git clang-format' in the repository."
        echo "You can also apply the patch directly:"
        echo "git apply @CMAKE_SOURCE_DIR@/.clang_format_diff.patch"
    fi
fi

# Filter Python files
python_files=()
for commit_file in ${commit_files[@]}
do
    if [[ $commit_file =~ .*\.py$ ]]; then
        python_files+=("${commit_file}")
    fi
done

# Python file checks
if [ ${#python_files[@]} -ne 0 ]; then
    # Use black to check Python file formatting, only if it is installed.
    @Python_EXECUTABLE@ -m black --version > /dev/null
    if [ $? -eq 0 ]; then
        @Python_EXECUTABLE@ -m black --check --quiet ${python_files[@]}
        if [ $? -ne 0 ]; then
            found_error=1
            printf "Found Python formatting errors.\n"
            printf "Please run 'black .' in the repository.\n"
        fi
    fi

    # Use isort to check python import order, only if it is installed.
    @Python_EXECUTABLE@ -m isort --version > /dev/null
    if [ $? -eq 0 ]; then
        @Python_EXECUTABLE@ -m isort --check-only --quiet ${python_files[@]}
        if [ $? -ne 0 ]; then
            found_error=1
            printf "Found unsorted Python imports.\n"
            printf "Please run 'isort .' in the repository.\n"
        fi
    fi
fi

###############################################################################
if [ "$found_error" -eq "1" ]; then
    exit 1
fi
