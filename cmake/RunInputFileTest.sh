#!/bin/sh -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Positional arguments to this script:
# - $1: executable name
# - $2: path to input file
# - $3: directory name
# - $4: space-separated list of expected exit codes
# - $5: "true" to check output files or "false" to skip the check
# - $6: "true" to check output files are present or "false" to skip
# - $7: additional command-line arguments forwarded to the executable
# - $8: extra files to copy

# Set up test directory
test_dir=@CMAKE_BINARY_DIR@/tests/InputFiles/$3
rm -rf $test_dir
mkdir -p $test_dir
cd $test_dir

input_file=$2
check_output_values=$5
check_output_present=$6

input_dir=`dirname $input_file`
for file in $8;
do
    cp $input_dir/$file $test_dir
done

# Run the executable
restart=
for expected_code in $4 ; do
    if [ -z "$restart" ] ; then
        @SPECTRE_TEST_RUNNER@ @CMAKE_BINARY_DIR@/bin/$1 --input-file \
                            $input_file $7
        exit_code=$?
        restart=0
    else
        if [ $exit_code -ne 2 ] ; then
            echo "Must restart after exit code 2" >&2
            exit 1
        fi
        @SPECTRE_TEST_RUNNER@ @CMAKE_BINARY_DIR@/bin/$1 \
            +restart Checkpoints/Checkpoint_$(printf %04d $restart)
        exit_code=$?
        restart=$(expr $restart + 1)
    fi
    if [ $exit_code -ne $expected_code ]; then
        echo "ERROR: Exited with ${exit_code} instead of ${expected_code}" >&2
        exit 1
    fi
done

# Check output and clean up
if [ "$check_output_values" = "true" ]; then
    @Python_EXECUTABLE@ @CMAKE_SOURCE_DIR@/tools/CheckOutputFiles.py \
        --input-file $input_file --run-directory $test_dir \
        --cmake-source-directory @CMAKE_SOURCE_DIR@ \
        --cmake-bin-directory @CMAKE_BINARY_DIR@ \
        || exit 1
fi
if [ "$check_output_present" = "true" ]; then
    @Python_EXECUTABLE@ -m spectre.tools.CleanOutput \
                      --output-dir $test_dir $input_file \
        || exit 1
fi
rm -rf $test_dir
