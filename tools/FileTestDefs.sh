# Distributed under the MIT License.
# See LICENSE.txt for details.

# This file defines the framework for running simple textual checks on
# files in the repo and defines a collection of standard checks.

# A check C consists of three functions:
# - `C`, which takes one filename as argument and should return true
#   if there is a problem with the file, and
# - `C_report`, which takes the list of bad files and should print a
#   message about them, and
# - `C_test`, which is used to test the check if the global $1 =
#   --test.
# The report function will not be called if there are no bad files.
# `C_test` should consist of several calls to the test_check function.

# Exit with a failure message
die() {
    [ -n "$1" ] && echo "$1" >&2 || echo "died" >&2
    exit 1
}

# Option to enable color in grep or the empty string if grep does not
# support color
color_option=''
if grep --help 2>&1 | grep -q -e --color ; then
  color_option='--color=auto'
fi

# Utility function for reporters that enables lots of decorators in grep
pretty_grep() {
    GREP_COLOR='1;37;41' grep --with-filename -n $color_option "$@"
}

# Utility functions for checks classifying a file based on its name
is_includible() { [[ $1 =~ \.hpp$ ]] || [[ $1 =~ \.tpp$ ]] ; }
is_c++() { [[ $1 =~ \.cpp$ ]] || [[ $1 =~ \.hpp$ ]] || [[ $1 =~ \.tpp$ ]] ; }

# Utility function for checks that returns false if the first argument
# matches any of the shell regexes passed as subsequent arguments.
whitelist() {
    local check pattern
    check=$1
    shift
    for pattern in "$@" ; do
        [[ ${check} =~ ${pattern} ]] && return 1
    done
    return 0
}

# Main driver.  Takes a list of checks as arguments and a list of
# filenames as null separated strings on its standard input.  Returns
# true if all the checks passed.
run_checks() {
    local check failures file files ret

    ret=0

    files=()
    while IFS= read -d '' -r file ; do
        files+=("${file}")
    done

    for check in "$@" ; do
        failures=()
        for file in "${files[@]}" ; do
            ${check} "${file}" && failures+=("${file}")
        done
        if [ ${#failures[@]} -ne 0 ] ; then
            ret=1
            ${check}_report "${failures[@]}"
            echo
        fi
    done

    return "${ret}"
}

# test_check pass|fail filename contents
test_check() {
    # check and failed are global variables
    local contents expected file tempdir
    [ $# -eq 3 ] || die "Wrong number of arguments"
    expected=$1
    file=$2
    contents=$3
    [ "${expected}" = pass ] || [ "${expected}" = fail ] || \
        die "Expected pass or fail, got '${expected}'"
    [[ "${file}" =~ ^/ ]] && die "Can't test with absolute path"

    # Delete the temporary directory if the script exits.  (This is
    # reset before leaving this function.)
    trap 'rm -rf "${tempdir}"' EXIT
    tempdir=$(mktemp -d)
    pushd "${tempdir}" >/dev/null
    mkdir -p "$(dirname "${file}")"
    printf '%s' "${contents}" >"${file}"
    if ${check} "${file}" ; then
        if [ "${expected}" != fail ] ; then
            echo "${check} unexpectedly failed on ${file}:"
            cat "${file}"
            failed=yes
        fi
    else
        if [ "${expected}" != pass ] ; then
            echo "${check} unexpectedly passed on ${file}:"
            cat "${file}"
            failed=yes
        fi
    fi
    rm -rf "${tempdir}"
    trap - EXIT
    return 0
}

# Run the specified tests.  Automatically run on the standard_checks
# if $1 is --test.
run_tests() {
    local check failed
    failed=no
    for check in "$@" ; do
        ${check}_test
    done
    [ "${failed}" != no ] && die "Tests failed"
    return 0
}


###### Standard checks ######
standard_checks=()

# Check for lines longer than 80 characters
long_lines() {
    is_c++ "$1" && grep '^[^#].\{80,\}' "$1" | grep -Ev 'https?://' | \
        grep -v '// IWYU pragma:' >/dev/null
}
long_lines_report() {
    echo "Found lines over 80 characters:"
    # This doesn't filter out URLs, but I can't think of a way to do
    # that without breaking the highlighting.  They only get printed
    # if there's another problem in the file.
    pretty_grep '^[^#].\{80,\}' "$@"
}
long_lines_test() {
    local ten=xxxxxxxxxx
    local eighty=${ten}${ten}${ten}${ten}${ten}${ten}${ten}${ten}
    test_check pass foo.cpp "${eighty}"$'\n'
    test_check fail foo.cpp "${eighty}x"$'\n'
    test_check fail foo.hpp "${eighty}x"$'\n'
    test_check fail foo.tpp "${eighty}x"$'\n'
    test_check pass foo.yaml "${eighty}x"$'\n'
    test_check pass foo.cpp "#include ${eighty}x"$'\n'
    test_check pass foo.cpp "// IWYU pragma: no_include ${eighty}x"$'\n'
    test_check pass foo.cpp "xxx http://${eighty}x"$'\n'
    test_check pass foo.cpp "xxx https://${eighty}x"$'\n'
}
standard_checks+=(long_lines)

# Check for files containing tabs
tabs() {
    whitelist "$1" '.png' &&
    grep -q -F $'\t' "$1"
}
tabs_report() {
    echo "Found tabs in the following files:"
    pretty_grep -F $'\t' "$@"
}
tabs_test() {
    test_check pass foo.cpp "x x"$'\n'
    test_check fail foo.cpp x$'\t'x$'\n'
}
standard_checks+=(tabs)

# Check for end-of-line spaces
trailing_space() {
    whitelist "$1" '.png' &&
    grep -q -E ' +$' "$1"
}
trailing_space_report() {
    echo "Found white space at end of line in the following files:"
    pretty_grep -E ' +$' "$@"
}
trailing_space_test() {
    test_check pass foo.cpp ' x'$'\n'
    test_check fail foo.cpp 'x '$'\n'
}
standard_checks+=(trailing_space)

# Check for carriage returns
carriage_returns() {
    whitelist "$1" '.png' &&
    grep -q -F $'\r' "$1"
}
carriage_returns_report() {
    echo "Found carriage returns in the following files:"
    # Skip highlighting because trying to highlight a carriage return
    # confuses some terminals.
    pretty_grep ${color_option:+--color=no} -F $'\r' "$@"
}
carriage_returns_test() {
    test_check pass foo.cpp 'x'
    test_check fail foo.cpp $'\r'
}
standard_checks+=(carriage_returns)

# Check for license file.
license() {
    whitelist "$1" \
              'cmake/FindCatch.cmake$' \
              'cmake/CodeCoverage.cmake$' \
              'cmake/CodeCoverageDetection.cmake$' \
              'cmake/FindLIBCXX.cmake$' \
              'cmake/FindPAPI.cmake$' \
              'cmake/FindPythonModule.cmake$' \
              'cmake/Findcppcheck.cmake$' \
              'cmake/Findcppcheck.cpp$' \
              'docs/config/footer.html' \
              'docs/config/header.html' \
              'docs/config/layout.xml' \
              'LICENSE' \
              'support/TeXLive/texlive.profile' \
              'tools/Iwyu/boost-all.imp$' \
              '.github/ISSUE_TEMPLATE.md' \
              '.github/PULL_REQUEST_TEMPLATE.md' \
              '.png' \
              '.svg' \
              '.clang-format$' && \
        ! grep -q "Distributed under the MIT License" "$1"
}
license_report() {
    echo "Did not find a license in these files:"
    printf '%s\n' "$@"
}
license_test() {
    test_check pass foo.cpp 'XXDistributed under the MIT LicenseXX'
    test_check fail foo.cpp ''
    test_check pass LICENSE ''
}
standard_checks+=(license)

# Check for tests using Catch's TEST_CASE instead of SPECTRE_TEST_CASE
test_case() {
    is_c++ "$1" && grep -q "^TEST_CASE" "$1"
}
test_case_reoprt() {
    echo "Found occurrences of TEST_CASE, must use SPECTRE_TEST_CASE:"
    pretty_grep "^TEST_CASE" "$@"
}
test_case_test() {
    test_check pass foo.cpp ''
    test_check pass foo.cpp 'SPECTRE_TEST_CASE()'
    test_check fail foo.cpp 'TEST_CASE()'
    test_check pass foo.yaml 'TEST_CASE()'
}
standard_checks+=(test_case)

# Check for tests using Catch's Approx, which has a very loose tolerance
catch_approx() {
    is_c++ "$1" && grep -q "Approx(" "$1"
}
catch_approx_report() {
    echo "Found occurrences of Approx, must use approx from"
    echo "tests/Unit/TestHelpers.hpp instead:"
    pretty_grep "Approx(" "$@"
}
catch_approx_test() {
    test_check pass foo.cpp ''
    test_check pass foo.cpp 'a == approx(b)'
    test_check fail foo.cpp 'a == Approx(b)'
    test_check pass foo.yaml 'a == Approx(b)'
}
standard_checks+=(catch_approx)

# Check for Doxygen comments on the same line as a /*!
doxygen_start_line() {
    is_c++ "$1" && grep -q '/\*\![^\n]' "$1"
}
doxygen_start_line_report() {
    echo "Found occurrences of bad Doxygen syntax: /*! STUFF:"
    pretty_grep -E '\/\*\!.*' "$@"
}
doxygen_start_line_test() {
    test_check pass foo.cpp ''
    test_check pass foo.cpp '  /*!'$'\n'
    test_check fail foo.cpp '  /*! '$'\n'
    test_check pass foo.yaml '  /*! '$'\n'
}
standard_checks+=(doxygen_start_line)

# Check for Ls because of a preference not to use it as short form for List
ls_list() {
    is_c++ "$1" && grep -q Ls "$1"
}
ls_list_report() {
    echo "Found occurrences of 'Ls', which is usually short for List:"
    pretty_grep Ls "$@"
}
ls_list_test() {
    test_check pass foo.cpp ''
    test_check pass foo.cpp ' FooList '
    test_check fail foo.cpp ' FooLs '
    test_check pass foo.yaml ' FooLs '
}
standard_checks+=(ls_list)

# Check for pragma once in all header files
pragma_once() {
    is_includible "$1" && \
        whitelist "$1" \
                  'tools/SpectrePch.hpp$' && \
        ! grep -q -x '#pragma once' "$1"
}
pragma_once_report() {
    echo "Did not find '#pragma once' in these header files:"
    printf '%s\n' "$@"
}
pragma_once_test() {
    test_check pass foo.cpp ''
    test_check fail foo.hpp ''
    test_check fail foo.tpp ''
    test_check pass foo.hpp '#pragma once'$'\n'
    test_check fail foo.hpp '//#pragma once'$'\n'
    test_check pass foo.hpp $'\n''#pragma once'$'\n\n'
}
standard_checks+=(pragma_once)

# Check for a newline at end of file
final_newline() {
    whitelist "$1" '.png' '.svg' &&
    # Bash strips trailing newlines from $() output
    [ "$(tail -c 1 "$1" ; echo x)" != $'\n'x ]
}
final_newline_report() {
    echo "No newline at end of file in:"
    printf '%s\n' "$@"
}
final_newline_test() {
    test_check pass foo.cpp $'\n'
    test_check fail foo.cpp ''
    test_check fail foo.cpp $'\n'x
}
standard_checks+=(final_newline)

# Check for enable_if and request replacing it with Requires
enable_if() {
    is_c++ "$1" && \
        whitelist "$1" \
                  'src/DataStructures/Tensor/Structure.hpp$' \
                  'src/IO/H5/File.hpp$' \
                  'src/Parallel/CharmMain.cpp$' \
                  'src/Utilities/PointerVector.hpp$' \
                  'src/Utilities/Requires.hpp$' \
                  'src/Utilities/TMPL.hpp$' \
                  'src/Utilities/TaggedTuple.hpp$' \
                  'tests/Unit/Utilities/Test_TypeTraits.cpp$' && \
        grep -q std::enable_if "$1"
}
enable_if_report() {
    echo "Found occurrences of 'std::enable_if', prefer 'Requires':"
    pretty_grep std::enable_if "$@"
}
enable_if_test() {
    test_check pass foo.cpp 'enable'
    test_check pass foo.cpp 'enable if'
    test_check pass foo.cpp 'enable_if'
    test_check fail foo.cpp 'std::enable_if'
}
standard_checks+=(enable_if)

# Check for struct TD and class TD asking to remove it
struct_td() {
    is_c++ "$1" && grep -q "\(struct TD;\|class TD;\)" "$1"
}
struct_td_report() {
    echo "Found 'struct TD;' or 'class TD;' which should be removed"
    pretty_grep "\(struct TD;\|class TD;\)" "$@"
}
struct_td_test() {
    test_check pass foo.cpp ''
    test_check fail foo.cpp 'struct TD;'
    test_check fail foo.cpp 'class TD;'
}
standard_checks+=(struct_td)

# Check for _details and details namespaces, request replacement with detail
namespace_details() {
    is_c++ "$1" && grep -q "\(_details\|namespace[[:space:]]\+details\)" "$1"
}
namespace_details_report() {
    echo "Found '_details' namespace, please replace with '_detail'"
    pretty_grep "\(_details\|namespace details\)" "$@"
}
namespace_details_test() {
    test_check pass foo.cpp ''
    test_check fail foo.cpp 'namespace details'
    test_check fail foo.cpp 'namespace    details'
    test_check fail foo.cpp 'namespace Test_details'
    test_check pass foo.cpp 'namespace Test_detail'
    test_check pass foo.cpp 'namespace detail'
    test_check pass foo.cpp 'details'
}
standard_checks+=(namespace_details)

[ "$1" = --test ] && run_tests "${standard_checks[@]}"

# True result for sourcing
:
