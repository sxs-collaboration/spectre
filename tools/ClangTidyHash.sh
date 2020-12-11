#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

if [ $# != 3 ] && [ $# != 4 ]; then
    echo "Wrong number of arguments passed to ClangTidyHash.sh, $#"
    echo "Expecting BUILD_DIR SOURCE_DIR HASH"
    echo "or BUILD_DIR SOURCE_DIR HASH NUM_THREADS_TO_RUN_ON"
    exit 1
fi

BUILD_DIR=$(realpath $1)
SOURCE_DIR=$(realpath $2)
HASH=$3

THREADING_FLAG="-j 1"
if [ $# == 4 ]; then
    THREADING_FLAG="-j $4"
fi

###############################################################################
cd $SOURCE_DIR
# Get list of non-deleted files
MODIFIED_FILES=()

for FILENAME in `git diff --name-only ${HASH} HEAD`
do
    if [ -f $FILENAME ] \
           && [ ${FILENAME: -4} == ".cpp" ] \
           && ! grep -q "FILE_IS_COMPILATION_TEST" $FILENAME; then
        MODIFIED_FILES+=("$FILENAME")
    fi
done

if [ ${#MODIFIED_FILES[@]} == 0 ]; then
   echo "No C++ files to check with clang-tidy"
   exit 0
fi

# Go to build directory and run clang-tidy
cd ${BUILD_DIR}

# Try to find run-clang-tidy-LLVM_VERSION in the path.
RUN_CLANG_TIDY_BIN=
for dir in $(echo ${PATH} | tr ':' '\n');
do
    if [ -d $dir ]; then
        RUN_CLANG_TIDY_BIN=`find $dir -name run-clang-tidy* | head -n 1`
        if [ ! -z "${RUN_CLANG_TIDY_BIN}" ]; then
            break
        fi
    fi
done

EXIT_CODE=0

# Check for run-clang-tidy, then run-clang-tidy-LLVM_VERSION, and
# if we can't find those, then just loop over the files one at a time.
RAN_CLANG_TIDY=no
for run_clang_tidy in run-clang-tidy run-clang-tidy.py "${RUN_CLANG_TIDY_BIN}" ;
do
    if command -v "${run_clang_tidy}" > /dev/null 2>&1; then
        ${run_clang_tidy} -quiet ${THREADING_FLAG} -p ${BUILD_DIR} \
                          ${MODIFIED_FILES[@]} || EXIT_CODE=1
        RAN_CLANG_TIDY=yes
        break
    fi
done
if [ ${RAN_CLANG_TIDY} == no ]; then
    for FILENAME in ${MODIFIED_FILES[@]};
    do
        printf "\nChecking file $FILENAME...\n"
        make clang-tidy FILE=${SOURCE_DIR}/${FILENAME} || EXIT_CODE=1
    done
fi

exit $EXIT_CODE
