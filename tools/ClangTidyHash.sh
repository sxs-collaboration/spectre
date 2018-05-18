#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

if [ $# != 3 ]; then
    echo "Wrong number of arguments passed to ClangTidyHash.sh"
    echo "Expecting BUILD_DIR SOURCE_DIR HASH"
    exit 1
fi

BUILD_DIR=$1
SOURCE_DIR=$2
HASH=$3

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

# Go to build directory and then run clang-tidy, writing output to file
cd ${BUILD_DIR}

for FILENAME in ${MODIFIED_FILES[@]}
do
    printf "\n\nRunning clang-tidy on file $FILENAME\n"
    make clang-tidy FILE=$SOURCE_DIR/${FILENAME}
done
