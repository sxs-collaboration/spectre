#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# This script can be invoked with the --test argument to run its unit
# tests instead of checking the repository.

top_level=$(git rev-parse --show-cdup) || exit 1
. "${top_level}tools/FileTestDefs.sh"

# redefine grep functions to run on the working directory
staged_grep() {
    grep "$@";
}
pretty_grep() {
    GREP_COLOR='1;37;41' grep --with-filename -n $color_option "$@"
}

##### CI checks #####
ci_checks=()

# Check for iostream header
iostream() {
    is_c++ "$1" && grep -q '#include <iostream>' "$1"
}
iostream_report() {
    echo "Found iostream header:"
    pretty_grep '#include <iostream>' "$@"
}
iostream_test() {
    test_check pass foo.cpp '#include <vector>'$'\n'
    test_check fail foo.cpp '#include <iostream>'$'\n'
}
ci_checks+=(iostream)

# Check for TmplDebugging header
tmpl_debugging() {
    is_c++ "$1" && grep -q '#include "Utilities/TmplDebugging.hpp"' "$1"
}
tmpl_debugging_report() {
    echo "Found Utilities/TmplDebugging header:"
    pretty_grep '#include "Utilities/TmplDebugging.hpp"' "$@"
}
tmpl_debugging_test() {
    test_check pass foo.cpp '#include <vector>'$'\n'
    test_check fail foo.cpp '#include "Utilities/TmplDebugging.hpp"'$'\n'
}
ci_checks+=(tmpl_debugging)

if [ "$1" = --test ] ; then
    run_tests "${ci_checks[@]}"
    exit 0
fi

# Exclude files that are generated, out of our control, etc.
if ! find . \
     -type f \
     ! -path "./.git/*" \
     ! -path "./docs/*" \
     ! -path "./build*" \
     ! -path '*/__pycache__/*' \
     ! -name '*.pyc' \
     ! -path "*.idea/*" \
     ! -name '*~' \
     ! -name "*.patch" \
     ! -path "./external/*" \
     ! -name deploy_key.enc \
     -print0 \
        | run_checks "${standard_checks[@]}" "${ci_checks[@]}"
then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    exit 1
fi
