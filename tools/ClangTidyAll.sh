#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

ALL_SOURCE_FILES=`find \
                  @CMAKE_SOURCE_DIR@/src \
                  @CMAKE_SOURCE_DIR@/tests \
                  -name "*.cpp"`

if [ ${#ALL_SOURCE_FILES[@]} == 0 ]; then
   echo "No C++ files found in repository"
   exit 0
fi

BUILD_DIR=@CMAKE_BINARY_DIR@

if [ $# != 0 ] && [ $# != 1 ]; then
    echo "Wrong number of arguments passed to ClangTidyAll.sh, $#"
    echo "Expecting none or just the number of threads to run on."
    exit 1
fi

THREADING_FLAG="-j 1"
if [ $# == 1 ]; then
    THREADING_FLAG="-j $1"
fi

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
                          ${ALL_SOURCE_FILES} || EXIT_CODE=1
        RAN_CLANG_TIDY=yes
        break
    fi
done
if [ ${RAN_CLANG_TIDY} == no ]; then
    for FILENAME in ${ALL_SOURCE_FILES};
    do
        printf "\nChecking file $FILENAME...\n"
        make clang-tidy FILE=${FILENAME} || EXIT_CODE=1
    done
fi

exit $EXIT_CODE
