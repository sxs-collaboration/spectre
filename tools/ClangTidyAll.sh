#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

ALL_SOURCE_FILES=`find \
                  @CMAKE_SOURCE_DIR@/src \
                  @CMAKE_SOURCE_DIR@/tests \
                  -name "*.cpp"`

for file in $ALL_SOURCE_FILES
do
    printf "\n\nChecking file: $file\n"
    make clang-tidy FILE=$file
done
