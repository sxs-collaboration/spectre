#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# All files, excluding generated things and files where formatting is
# outside our control
all_files() {
    find \
        -type f \
        ! -path "./.git/*" \
        ! -path "./docs/*" \
        ! -path "./build*" \
        ! -path '*/__pycache__/*' \
        ! -path "*.idea/*" \
        ! -name "*.patch" \
        ! -name deploy_key.enc
}

# All non-generated cpp and hpp files
all_c++_files() {
    find \
        -type f \
        ! -path "./build*" \
        -name '*.[ch]pp'
}

# All non-generated hpp files
all_hpp_files() {
    find \
        -type f \
        ! -path "./build*" \
        -name '*.hpp'
}

###############################################################################
# Add grep colors if available
color_option=''
if grep --help 2>&1 | grep -q -e --color
then
  color_option='--color=auto'
fi

pretty_grep() {
    GREP_COLOR='1;37;41' grep --with-filename -n $color_option "$@"
}

found_error=0

###############################################################################
# Check for lines longer than 80 characters
found_long_lines=`all_c++_files | xargs grep --with-filename -n '.\{81,\}'`
if [[ $found_long_lines != "" ]]; then
    echo "Found lines over 80 characters:"
    echo "$found_long_lines"
    echo
    found_error=1
fi

###############################################################################
# Check for iostream header
found_iostream=`
all_c++_files | xargs grep --with-filename -n '#include <iostream>'`
if [[ $found_iostream != "" ]]; then
    echo "Found iostream header in files:"
    echo "$found_iostream"
    echo
    found_error=1
fi

###############################################################################
# Find lines that have tabs in them and block them from being committed
found_tabs_files=`all_files | xargs grep -lF $'\t'`
if [[ $found_tabs_files != "" ]]; then
    echo "Found tabs in the following files:"
    pretty_grep -F $'\t' $found_tabs_files
    echo
    found_error=1
fi

###############################################################################
# Find files that have white space at the end of a line and block them
found_spaces_files=`all_files | xargs grep -lE ' +$'`
if [[ $found_spaces_files != "" ]]; then
    echo "Found white space at end of line in the following files:"
    pretty_grep -E ' +$' $found_spaces_files
    echo
    found_error=1
fi

###############################################################################
# Check for carriage returns
found_carriage_return_files=`all_files | xargs grep -lF $'\r'`
if [[ $found_carriage_return_files != "" ]]; then
    echo "Found carriage returns in the following files:"
    # Skip highlighting because trying to highlight a carriage return
    # confuses some terminals.
    pretty_grep ${color_option:+--color=no} -F $'\r' \
                $found_carriage_return_files
    echo
    found_error=1
fi

###############################################################################
# Check for license file. We need to ignore a few files that shouldn't have
# the license
files_without_license=()
for file in $(all_files | xargs grep -L "Distributed under the MIT License")
do
    case "${file}" in
        *cmake/FindLIBCXX.cmake) ;;
        *cmake/FindPAPI.cmake) ;;
        *cmake/CodeCoverageDetection.cmake) ;;
        *cmake/CodeCoverage.cmake) ;;
        *cmake/Findcppcheck.cmake) ;;
        *cmake/Findcppcheck.cpp) ;;
        *cmake/FindCatch.cmake) ;;
        *cmake/FindPythonModule.cmake) ;;
        *LICENSE.*) ;;
        *.clang-format) ;;
        *) files_without_license+=("${file}")
    esac
done
if [[ ${#files_without_license[@]} -ne 0 ]]; then
    echo "Did not find a license in these files:"
    printf '%s\n' "${files_without_license[@]}"
    echo
    found_error=1
fi

###############################################################################
# Check for tests using Catch's TEST_CASE instead of SPECTRE_TEST_CASE
found_test_case=`all_c++_files | xargs grep -l "^TEST_CASE"`
if [[ $found_test_case != "" ]]; then
    echo "Found occurrences of TEST_CASE, must use SPECTRE_TEST_CASE:"
    pretty_grep "^TEST_CASE" $found_test_case
    echo
    found_error=1
fi

###############################################################################
# Check for tests using Catch's Approx, which has a very loose tolerance
found_bad_approx=`all_c++_files | xargs grep -l "Approx("`
if [[ $found_bad_approx != "" ]]; then
    printf "Found occurrences of Approx, must use approx from " \
           "SPECTRE_ROOT/tests/Unit/TestHelpers.hpp instead:\n"
    pretty_grep "Approx(" $found_bad_approx
    echo
    found_error=1
fi

###############################################################################
# Check for Doxygen comments on the same line as a /*!
found_bad_doxygen_syntax=`all_c++_files | xargs grep -l '/\*\![^\n]'`
if [[ $found_bad_doxygen_syntax != "" ]]; then
    printf "Found occurrences of bad Doxygen syntax: /*! STUFF\n"
    pretty_grep -E '\/\*\!.*' $found_bad_doxygen_syntax
    echo
    found_error=1
fi

###############################################################################
# Check for Ls because of a preference not to use it as short form for List
found_incorrect_list_name=`all_c++_files | xargs grep -l 'Ls'`
if [[ $found_incorrect_list_name != "" ]]; then
    printf "Found occurrences of 'Ls', which is usually short for List\n"
    pretty_grep 'Ls' $found_incorrect_list_name
    echo
    found_error=1
fi

###############################################################################
# Check for pragma once in all hearder files
found_no_pragma_once=`all_hpp_files | xargs grep -L "^#pragma once$"`
if [[ $found_no_pragma_once != "" ]]; then
    printf "Did not find '#pragma once' in these header files:\n"
    echo "$found_no_pragma_once"
    echo
    found_error=1
fi

if [ "$found_error" -eq "1" ]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    exit 1
fi
