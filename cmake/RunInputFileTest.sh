#!/bin/sh -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Positional arguments to this script:
# - $1: executable name
# - $2: path to input file
# - $3: directory name
# - $4: expected exit code
# - $5: "true" to check output files or "false" to skip the check
# - $6: additional command-line arguments forwarded to the executable

# Set up test directory
test_dir=@CMAKE_BINARY_DIR@/tests/InputFiles/$3
rm -rf $test_dir
mkdir -p $test_dir
cd $test_dir

# Run the executable
@SPECTRE_TEST_RUNNER@ @CMAKE_BINARY_DIR@/bin/$1 --input-file $2 ${6}
exit_code=$?
if [ $exit_code -ne $4 ]; then
    exit 1
fi

# Check output and clean up
if [ "$5" = "true" ]; then
    @Python_EXECUTABLE@ @CMAKE_SOURCE_DIR@/tools/CheckOutputFiles.py \
        --input-file $2 --run-directory $test_dir \
        || exit 1
fi
@Python_EXECUTABLE@ -m spectre.tools.CleanOutput \
    --output-dir $test_dir $2 \
    || exit 1
rm -rf $test_dir
