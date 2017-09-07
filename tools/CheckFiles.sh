#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

###############################################################################
# Add grep colors if available
color_option=''
if grep --help 2>&1 | grep -q -e --color
then
  color_option='--color=auto'
fi

found_error=0

###############################################################################
# Check for lines longer than 80 characters
found_long_lines=`
find ./ -type f -name '*.[ch]pp' -not -path './build*' \
    | xargs grep --with-filename -n '.\{81,\}'`
if [[ $found_long_lines != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    echo "Found lines over 80 characters:"
    echo "$found_long_lines"
    found_error=1
fi

###############################################################################
# Check for iostream header
found_iostream=`
find ./ -type f -name '*.[ch]pp' -not -path './build*' \
    | xargs grep --with-filename -n '#include <iostream>'`
if [[ $found_iostream != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    echo "Found iostream header in files:"
    echo "$found_iostream"
    found_error=1
fi

###############################################################################
# Find lines that have tabs in them and block them from being committed
found_tabs_files=`
find ./ -type f               \
     ! -name "*.patch"        \
     ! -path "./docs/*"       \
     ! -path "./.git/*"       \
     ! -path "./build*"       \
    | xargs grep -E '^.*'$'\t'`
if [[ $found_tabs_files != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    echo "Found tabs in the following  files:"
    echo "$found_tabs_files" | GREP_COLOR='1;37;41' grep -F $'\t' $color_option
    found_error=1
fi

###############################################################################
# Find files that have white space at the end of a line and block them
found_spaces_files=`
find ./ -type f             \
     ! -name "*.patch"      \
     ! -path "./docs/*"     \
     ! -path "./.git/*"     \
     ! -path "./build*"     \
    | xargs grep -E '^.* +$'`
echo
if [[ $found_spaces_files != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    echo "Found white space at end of line in the following files:"
    echo "$found_spaces_files" | \
        GREP_COLOR='1;37;41' grep -E ' +$' $color_option
    found_error=1
fi

###############################################################################
# Check for carriage returns
found_carriage_return_files=`
find ./ -type f                 \
     ! -path "*.git*"           \
     ! -path "./build*"       \
    | xargs grep -E '^\+.*'$'\r'`
if [[ $found_carriage_return_files != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    echo "Found carriage returns in the following files:"
    echo "$found_carriage_return_files" | \
        GREP_COLOR='1;37;41' grep -E '\r+$' $color_option
    found_error=1
fi

###############################################################################
# Check for license file. We need to ignore a few files that shouldn't have
# the license
files_without_license=`
find ./ -type f                                                              \
     ! -path "*cmake/FindLIBCXX.cmake"                                       \
     ! -path "*cmake/FindPAPI.cmake"                                         \
     ! -path "*cmake/CodeCoverageDetection.cmake"                            \
     ! -path "*cmake/CodeCoverage.cmake"                                     \
     ! -path "*cmake/Findcppcheck.cmake"                                     \
     ! -path "*cmake/Findcppcheck.cpp"                                       \
     ! -path "*cmake/FindCatch.cmake"                                        \
     ! -path "*cmake/FindPythonModule.cmake"                                 \
     ! -path "./src/Utilities/Gsl.hpp"                                       \
     ! -path "*.git/*"                                                       \
     ! -path "*.idea/*"                                                      \
     ! -path "*docs/config/*"                                                \
     ! -path "*.travis/deploy_key*"                                          \
     ! -name "*.patch"                                                       \
     ! -name "*LICENSE.*"                                                    \
     ! -name "*.clang-format"                                                \
     ! -path "./build*"                                                      \
    | xargs grep -L "^.*Distributed under the MIT License"`
if [[ $files_without_license != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    echo "Did not find a license in these files:"
    echo "$files_without_license"
    found_error=1
fi

###############################################################################
# Check for tests using Catch's TEST_CASE instead of SPECTRE_TEST_CASE
found_test_case=`
find ./ -type f -name '*.[ch]pp' -not -path './build*' \
    | xargs grep --with-filename -n "^TEST_CASE"`
if [[ $found_test_case != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    echo "Found occurrences of TEST_CASE, must use SPECTRE_TEST_CASE:"
    echo "$found_test_case"
    found_error=1
fi

###############################################################################
# Check for tests using Catch's Approx, which has a very loose tolerance
found_bad_approx=`
find ./ -type f -name '*.[ch]pp' -not -path './build*' \
    | xargs grep --with-filename -n "Approx("`
if [[ $found_bad_approx != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    printf "Found occurrences of Approx, must use approx from " \
           "SPECTRE_ROOT/tests/Unit/TestHelpers.hpp instead:\n"
    echo "$found_bad_approx"
    found_error=1
fi

if [ "$found_error" -eq "1" ]; then
    exit 1
fi
