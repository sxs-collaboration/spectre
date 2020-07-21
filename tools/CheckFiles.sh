#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# This script can be invoked with the --test argument to run its unit
# tests instead of checking the repository.

# In addition to sourcing various helper functions, source the standard file
# checks, i.e., the file checks that are performed in the git commit hook AND
# by CI testing.
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
# (Checked in CI only to allow local debugging-related commits)
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
# (Checked in CI only to allow local debugging-related commits)
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

# Check that every C++ file is listed in the directory's CMakeLists.txt
# (Checked in CI because this involves comparisons between files and may
#  be too slow for a fluid git-hook user experience)
#
# We check for C++ files in the src directory only, and even there we omit any
# directories named Executables or Python. This is because we currently only
# expect the main SpECTRE libraries from src to list all their C++ files.
# The CMakeLists for Python bindings, executables, and tests will be updated
# in the future.
# We also omit special C++ files we know shouldn't be in their CMakeLists.txt.
check_cmakelists_for_missing_cpp() {
    local dir base cmakelists
    dir=$(dirname $1)
    base=$(basename $1)
    cmakelists="${dir}/CMakeLists.txt"
    is_c++ "$1" \
      && whitelist "$dir" \
                   'docs' \
                   'tests/*' \
                   'tools' \
                   'Executables' \
                   'Python' \
      && whitelist "$1" \
                   'src/Informer/InfoAtCompile.cpp$' \
                   'src/Informer/InfoAtLink.cpp$' \
      && [ -f $cmakelists ] \
      && [ $(grep -L "^  $base" $cmakelists) ]
}
check_cmakelists_for_missing_cpp_report() {
    local file dir base cmakelists
    echo "Found C++ files not in CMakeLists.txt:"
    for file in "$@"; do
        dir=$(dirname $file)
        base=$(basename $file)
        cmakelists="${dir}/CMakeLists.txt"
        echo "$base should be added to $cmakelists"
    done
}
check_cmakelists_for_missing_cpp_test() {
    # This check relies on comparisons between different files, which makes
    # it cumbersome to test. We omit the test.
    :
}
ci_checks+=(check_cmakelists_for_missing_cpp)

# Check CMakeLists.txt contain no spurious C++ files
# (Checked in CI because this involves comparisons between files and may
#  be too slow for a fluid git-hook user experience)
#
# Check for CMakeLists lines that are [two spaces][anything][.?pp]
# Then we check that the filename has no slashes (this would indicate a file
# in a subdirectory), and that the corresponding file exists.
check_cmakelists_for_extra_cpp() {
    local dir match matches
    dir=$(dirname $1)
    if [[ $1 =~ CMakeLists\.txt$ ]] && whitelist "$dir" 'docs' \
                                                        'tests' \
                                                        'tools' \
                                                        'Executables' \
                                                        'Python'; then
        matches=$(grep -E "^  .*\.[cht]pp" $1)
        for match in $matches; do
            # Special case: the Informer CMakeLists includes a special C++ file
            [[ $match == '${CMAKE_BINARY_DIR}/Informer/InfoAtCompile.cpp' ]] \
              && continue
            [[ $match =~ '/' ]] || [[ ! -f "${dir}/$match" ]] && return 0
        done
    fi
    return 1
}
check_cmakelists_for_extra_cpp_report() {
    local file dir match matches
    echo "Found spurious C++ files in CMakeLists.txt:"
    for file in "$@"; do
        dir=$(dirname $file)
        matches=$(grep -E "^  .*\.[cht]pp" $file)
        for match in $matches; do
            [[ $match == '${CMAKE_BINARY_DIR}/Informer/InfoAtCompile.cpp' ]] \
              && continue
            [[ $match =~ '/' ]] || [[ ! -f "${dir}/$match" ]] \
              && echo "$match should be removed from $file"
        done
    done
}
check_cmakelists_for_extra_cpp_test() {
    # This check relies on comparisons between different files, which makes
    # it cumbersome to test. We omit the test.
    :
}
ci_checks+=(check_cmakelists_for_extra_cpp)

if [ "$1" = --test ] ; then
    run_tests "${ci_checks[@]}"
    exit 0
fi

# Exclude files that are generated, out of our control, etc.
if ! find . \
     -type f \
     ! -path './.git/*' \
     ! -path './build*' \
     ! -path './docs/*' \
     ! -path './external/*' \
     ! -path '*.idea/*' \
     ! -name '*.patch' \
     ! -name '*.pyc' \
     ! -path '*/__pycache__/*' \
     ! -name '*~' \
     ! -name deploy_key.enc \
     -print0 \
        | run_checks "${standard_checks[@]}" "${ci_checks[@]}"
then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    exit 1
fi
