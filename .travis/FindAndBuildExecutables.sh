#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# This script searches for executables that the input
# file tests depend on and then builds them.
# The first argument passed in is the directory in
# which to search
input_files=`find $1 -name '*.yaml'`

executables=()
for input_file in $input_files; do
    executables+=(`grep 'Executable' $input_file \
                   | sed s/\#[[:space:]]*Executable:[[:space:]]*//g`)
done
echo "Found: ${executables[@]}"
echo "Removing duplicate executables"
executables=`echo "${executables[@]}" | xargs -n1 | sort -u`

for executable in $executables; do
    echo "Building $executable"
    make $executable
done
