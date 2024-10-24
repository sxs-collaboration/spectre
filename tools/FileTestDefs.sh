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
  color_option='--color=always'
fi

# Utility that uses grep on the staged version of the file specified by the last
# argument does not work with multiple files as argument
staged_grep() {
    # git show ":./path/to/file" shows the content of the file as it
    # appears in the staging area
    git show ":./${@: -1}" | grep "${@:1:$(($#-1))}"
}

# Utility function for reporters that enables lots of decorators in
# grep.  Works like staged_grep, except that it can take multiple file
# arguments like real grep.  Accepts the same arguments as grep except
# that any option taking an argument must use the --opt=arg form
# instead of the short form.  Additionally, the order of options is
# restricted as compared to real grep: only filename arguments are
# allowed to follow the pattern.
pretty_grep() {
    local -a non_file_args
    local color file file_prefix
    color=${color_option}
    # This loop extracts all the flags and the pattern into
    # non_file_args, leaving the filenames in $@.
    while [ $# -gt 0 ] ; do
        case "$1" in
            # "--" indicates the end of command line switches, so the
            # following argument is the pattern and the remainder are
            # file names.
            --)
                non_file_args+=("$1" "$2")
                shift 2
                break
                ;;
            --color=*)
                color=$1
                shift
                ;;
            -*)
                non_file_args+=("$1")
                shift
                ;;
            *)
                # This is the pattern
                non_file_args+=("$1")
                shift
                break
                ;;
        esac
    done

    for file in "$@" ; do
        if [ -n "${color}" ] && [ "${color}" != "--color=no" ] ; then
            printf -v file_prefix "\033[0;35m%s\033[0m:" "${file}"
        else
            file_prefix=${file}:
        fi
        git show ":./${file}" | \
            GREP_COLOR='1;37;41' GREP_COLORS='mt=1;37;41' \
            grep -n $color "${non_file_args[@]}" | \
            sed "s|^|${file_prefix}|"
    done
}

# Utility functions for checks classifying a file based on its name
is_includible() { [[ $1 =~ \.hpp$ ]] || [[ $1 =~ \.tpp$ ]] ; }
is_c++() { [[ $1 =~ \.cpp$ ]] || [[ $1 =~ \.hpp$ ]] || [[ $1 =~ \.tpp$ ]] ; }
is_c++_or_python() { [[ $1 =~ \.cpp$ ]] || [[ $1 =~ \.hpp$ ]] \
                         || [[ $1 =~ \.tpp$ ]] || [[ $1 =~ \.py$ ]] ; }

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

# check_nolint [-q] lint file...
#
# Check for NOLINT comments in the supplied files matching the
# supplied `lint` string escaping the lines from standard input.
# Echos any non-escaped lines unless '-q' is passed as the first
# argument.  Returns 1 if everything was NOLINTED, 0 otherwise.
#
# The lines to match must not contain color sequences, because they
# are too hard to parse.  Pass ${color_option:+--color=no} to pretty_grep.
check_nolint() {
    local quiet=no
    if [ "$1" = "-q" ] ; then
        quiet=yes
        shift
    fi
    [ $# -ge 2 ] || die "Wrong number of arguments"
    perl -e '
use strict;
my ($quiet, $lint, @files) = @ARGV;
my $result = 1;
while (my $match = <STDIN>) {
  my $found = 0;
  my ($match_file, $match_lineno, $match_text) =
    $match =~ m#^([\w/.]+):(\d+):(.*)#;
  $match_text //= $match;
  chomp $match_text;
  foreach my $file (@files) {
    next if defined $match_file and $match_file ne $file;
    open F, $file or die;
    my $escaped = 0;
    while(<F>) {
      chomp;
      m# NOLINT(NEXTLINE)?\((?:|.*,)\Q$lint\E(?:|,.*)\)# and
        $escaped = $1 eq "NEXTLINE" ? 2 : 1;
      if ((not defined $match_lineno or $match_lineno == $.) and
           $_ eq $match_text) {
        $found = 1;
        if ($escaped != 1) {
          $result = 0;
          print $match unless $quiet eq "yes";
        }
      }
      --$escaped if $escaped;
    }
    die "Did not find line: $match" unless $found;
    close F;
  }
}
exit $result;
' "${quiet}" "$@"
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
            echo
            failed=yes
        fi
    else
        if [ "${expected}" != pass ] ; then
            echo "${check} unexpectedly passed on ${file}:"
            cat "${file}"
            echo
            failed=yes
        fi
    fi
    rm -rf "${tempdir}"
    popd >/dev/null
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
# Some patterns are allowed to exceed the line limit. Notes:
# - Black Python formatting prefers to end single-line docstrings on the same
#   line even if the '"""' exceeds the line limit.
# - Black also prefers to avoid backslashes even if that means a long import
#   statements exceeds the line limit (in case breaking the line with
#   parentheseses is not possible).
long_lines_exclude() {
    grep -Ev 'https?://' | \
        grep -v 'mailto:' | \
        grep -v '// NOLINT' | \
        grep -v '\\snippet' | \
        grep -v '\\image' | \
        grep -v 'a href=' | \
        grep -v '"""' | \
        grep -v 'import' | \
        grep -v '\\link' | \
        grep -v '\\endlink'
}
long_lines() {
    whitelist "$1" \
              '.cmake$' \
              '.css$' \
              '.h5$' \
              '.html$' \
              '.ipynb$' \
              '.js$' \
              '.json$' \
              '.min.js$' \
              '.mplstyle$' \
              '.patch' \
              '.serializations' \
              '.svg' \
              '.travis.yml$' \
              '.xml$' \
              '.xmf$' \
              'CITATION.cff' \
              'CMakeLists.txt$' \
              'Doxyfile.in$' \
              'containers/Dockerfile.buildenv$' \
              'containers/Dockerfile.travis$' \
              'docs/DevGuide/Travis.md' \
              'docs/MainSite/Main.md' \
              'docs/Tutorials/CCE.md' \
              'docs/Tutorials/ParallelExecutable/Tutorials.md' \
              'docs/config/MathJax.js' \
              'eos.*' \
              'external/Licenses/.*License.txt' \
              'RotatingStarId.dat$' \
              'RotatingStarId_Hybrid.dat$' \
              'tools/CheckFiles.sh$' && \
        staged_grep '^[^#].\{80,\}' "$1" | long_lines_exclude >/dev/null
}
long_lines_report() {
    echo "Found lines over 80 characters:"
    pretty_grep '^[^#].\{80,\}' "$@" | long_lines_exclude
}
long_lines_test() {
    local ten=xxxxxxxxxx
    local eighty=${ten}${ten}${ten}${ten}${ten}${ten}${ten}${ten}
    test_check pass foo.cpp "${eighty}"$'\n'
    test_check fail foo.cpp "${eighty}x"$'\n'
    test_check fail foo.hpp "${eighty}x"$'\n'
    test_check fail foo.tpp "${eighty}x"$'\n'
    test_check fail foo.yaml "${eighty}x"$'\n'
    test_check pass foo.cmake "${eighty}x"$'\n'
    test_check pass foo.cpp "#include ${eighty}x"$'\n'
    test_check pass foo.cpp "xxx http://${eighty}x"$'\n'
    test_check pass foo.cpp "xxx https://${eighty}x"$'\n'
    test_check pass foo.cpp "linted;  // NOLINT(${eighty})"$'\n'
    test_check pass foo.cpp "// NOLINTNEXTLINE(${eighty})"$'\n'
    test_check pass foo.cpp "// \\snippet ${eighty}"$'\n'
    test_check pass foo.cpp "// \\image ${eighty}"$'\n'
    test_check pass foo.cpp "// \\link ${eighty}"$'\n'
    test_check pass foo.cpp "// \\endlink ${eighty}"$'\n'
}
standard_checks+=(long_lines)

# Check for lrtslock header
lrtslock() {
    is_c++ "$1" && staged_grep -q '#include <lrtslock.h>' "$1"
}
lrtslock_report() {
    echo "Found lrtslock header (include converse.h instead):"
    pretty_grep '#include <lrtslock.h>' "$@"
}
lrtslock_test() {
    test_check pass foo.cpp '#include <vector>'$'\n'
    test_check fail foo.cpp '#include <lrtslock.h>'$'\n'
}
standard_checks+=(lrtslock)

# Check for boost::optional headers (we favor std::optional)
# (see related check for boost/none header below)
boost_optional() {
    is_c++ "$1" && staged_grep -q '#include <boost/optional.*>' "$1"
}
boost_optional_report() {
    echo "Found boost::optional header (use std::optional instead):"
    pretty_grep '#include <boost/optional.*>' "$@"
}
boost_optional_test() {
    test_check pass foo.cpp '#include <optional>'$'\n'
    test_check pass foo.cpp '#include <boost/none.hpp>'$'\n'
    test_check fail foo.cpp '#include <boost/optional.hpp>'$'\n'
    test_check fail foo.cpp '#include <boost/optional/optional_io.hpp>'$'\n'
}
standard_checks+=(boost_optional)

# Check for boost/none header
# (see related check for boost/optional* headers above)
boost_none() {
    is_c++ "$1" && staged_grep -q '#include <boost/none.hpp>' "$1"
}
boost_none_report() {
    echo "Found boost/none.hpp header:"
    pretty_grep '#include <boost/none.hpp>' "$@"
}
boost_none_test() {
    test_check pass foo.cpp '#include <optional>'$'\n'
    test_check pass foo.cpp '#include <boost/optional.hpp>'$'\n'
    test_check fail foo.cpp '#include <boost/none.hpp>'$'\n'
}
standard_checks+=(boost_none)

# Check for files containing tabs
tabs() {
    whitelist "$1" '.h5' '.ico' '.jpg' '.png' '.patch' &&
    staged_grep -q -F $'\t' "$1"
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
    whitelist "$1" '.h5' '.ico' '.jpg' '.png' '.patch' 'eos.*' &&
    staged_grep -q -E ' +$' "$1"
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
    whitelist "$1" '.h5' '.ico' '.jpg' '.png' &&
    staged_grep -q -F $'\r' "$1"
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
              '.clang-format$' \
              '.clang-tidy$' \
              '.github/ISSUE_TEMPLATE.md' \
              '.github/PULL_REQUEST_TEMPLATE.md' \
              '.h5' \
              '.ico' \
              '.jpg' \
              '.json' \
              '.nojekyll' \
              '.png' \
              '.serializations' \
              '.svg' \
              '.patch' \
              '.xmf' \
              'LICENSE' \
              'citation.bib' \
              'cmake/CodeCoverage.cmake$' \
              'cmake/CodeCoverageDetection.cmake$' \
              'cmake/FindCatch.cmake$' \
              'cmake/FindLIBCXX.cmake$' \
              'cmake/FindPAPI.cmake$' \
              'cmake/FindPythonModule.cmake$' \
              'docs/config/doxygen-awesome-sidebar-only.css' \
              'docs/config/doxygen-awesome.css' \
              'docs/config/doxygen-awesome-fragment-copy-button.js' \
              'docs/config/doxygen-awesome-interactive-toc.js' \
              'docs/config/doxygen-awesome-paragraph-link.js' \
              'docs/config/footer.html' \
              'docs/config/header.html' \
              'docs/config/layout.xml' \
              'docs/config/MathJax.js$' \
              'external/*' \
              'eos.*' \
              'RotatingStarId.dat$' \
              'RotatingStarId_Hybrid.dat$' \
              'support/TeXLive/texlive.profile' && \
        ! staged_grep -q "Distributed under the MIT License" "$1"
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
    is_c++ "$1" && staged_grep -q "^TEST_CASE" "$1"
}
test_case_report() {
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
    is_c++ "$1" && staged_grep -q "Approx(" "$1"
}
catch_approx_report() {
    echo "Found occurrences of Approx, must use approx from"
    echo "tests/Unit/Framework/TestingFramework.hpp instead:"
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
    is_c++ "$1" && staged_grep -q '/\*![^\n]' "$1"
}
doxygen_start_line_report() {
    echo "Found occurrences of bad Doxygen syntax: /*! STUFF:"
    pretty_grep -E '\/\*!.*' "$@"
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
    is_c++ "$1" && staged_grep -q Ls "$1"
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
                  'tools/SpectrePch.hpp$' \
                  'tools/BlazeExceptions.hpp$' && \
        ! staged_grep -q -x '#pragma once' "$1"
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

# Check for 'return Py_None;' in all C++ files
py_return_none() {
    is_c++ "$1" && \
        whitelist "$1" && \
        staged_grep -q -x '.*return Py_None;.*' "$1"
}
py_return_none_report() {
    echo "Found 'return Py_None;' in files. Use Py_RETURN_NONE instead."
    pretty_grep ".*return Py_None;.*" "$@"
}
py_return_none_test() {
    test_check pass foo.cpp ''
    test_check pass foo.hpp ''
    test_check pass foo.tpp ''
    test_check fail foo.hpp '  return Py_None;'$'\n'
    test_check fail foo.cpp '  return Py_None;'$'\n'
    test_check fail foo.tpp '  return Py_None;'$'\n'
    test_check fail foo.hpp '//return Py_None;'$'\n'
    test_check fail foo.cpp '//return Py_None;'$'\n'
    test_check fail foo.tpp '//return Py_None;'$'\n'
    test_check fail foo.hpp '  return Py_None; '$'\n'
    test_check fail foo.cpp '  return Py_None; '$'\n'
    test_check fail foo.tpp '  return Py_None; '$'\n'
    test_check pass foo.hpp '//return Py_None'$'\n'
    test_check pass foo.cpp '//return Py_None'$'\n'
    test_check pass foo.tpp '//return Py_None'$'\n'
}
standard_checks+=(py_return_none)

# Check for a newline at end of file
final_newline() {
    whitelist "$1" '.h5' '.ico' '.jpg' '.nojekyll' '.png' '.svg' &&
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
                  'src/Utilities/Requires.hpp$' \
                  'src/Utilities/TMPL.hpp$' \
                  'src/Utilities/TaggedTuple.hpp$' \
                  'tests/Unit/Utilities/Test_TypeTraits.cpp$' && \
        staged_grep -q std::enable_if "$1"
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

# Check for noexcept
noexcept() {
    whitelist "$1" \
              "src/Utilities/StdHelpers/Bit.hpp" \
              'src/Parallel/StaticSpscQueue.hpp' \
              "src/Parallel/NodeLock..pp$" \
              "src/Evolution/DiscontinuousGalerkin/\
AtomicInboxBoundaryData..pp$" && \
        is_c++ "$1" && \
        staged_grep -q 'noexcept ' "$1"
}
noexcept_report() {
    echo "Found occurrences of 'noexcept', please remove."
    pretty_grep noexcept "$@"
}
noexcept_test() {
    test_check pass foo.cpp ''
    test_check pass foo.hpp ''
    test_check pass foo.tpp ''
    test_check fail foo.hpp 'noexcept '
    test_check fail foo.cpp 'noexcept '
    test_check fail foo.tpp 'noexcept '
    test_check pass foo.hpp 'noexcept(true)'
    test_check pass foo.cpp 'noexcept(true)'
    test_check pass foo.tpp 'noexcept(true)'
}
standard_checks+=(noexcept)

# Check for mutable
mutable() {
    # We talk about mutable stuff a lot, so try to avoid flagging
    # occurrences in comments and strings.  Checking for end-of-line
    # comments and balanced quotes isn't worth it.  "mutable" is also
    # fine for mutable lambdas, but those are rare and hard to detect
    # with regex, so they can just be NOLINTed.
    is_c++ "$1" && \
        staged_grep ' mutable ' "$1" | \
            check_nolint spectre-mutable "$1" | \
            grep -v -E -q '^ *(//|\* )|".* mutable '
}
mutable_report() {
    echo "Found occurrences of 'mutable'.  Please discuss with the core devs."
    pretty_grep ${color_option:+--color=no} ' mutable ' "$@" | \
        check_nolint spectre-mutable "$@" | \
        grep -v -E '^([^:]*:){2} *(//|\* )|".* mutable '
}
mutable_test() {
    test_check pass foo.cpp ''
    test_check pass foo.hpp ''
    test_check pass foo.tpp ''
    test_check fail foo.hpp ' mutable '
    test_check fail foo.cpp ' mutable '
    test_check fail foo.tpp ' mutable '
    test_check pass foo.hpp ' mutable {  // NOLINT(spectre-mutable)'
    test_check pass foo.cpp ' mutable {  // NOLINT(spectre-mutable)'
    test_check pass foo.tpp ' mutable {  // NOLINT(spectre-mutable)'
    test_check fail foo.cpp ' mutable {  // NOLINT(notspectre-mutable)'
    test_check pass foo.cpp \
               '// NOLINTNEXTLINE(spectre-mutable)'$'\n'' mutable {'
    test_check pass foo.cpp '// mutable '
    test_check pass foo.cpp ' * mutable ' # Probably a multiline comment
    test_check pass foo.cpp '  // mutable '
    test_check pass foo.cpp ' "something mutable something" '
    test_check fail foo.cpp ' mutable // mutable '
}
standard_checks+=(mutable)

# Check for struct TD and class TD asking to remove it
struct_td() {
    is_c++ "$1" && staged_grep -q "\(struct TD;\|class TD;\)" "$1"
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
    is_c++ "$1" && \
        staged_grep -q "\(_details\|namespace[[:space:]]\+details\)" "$1"
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

# Check for .cpp includes in cpp, hpp, and tpp files
prevent_cpp_includes() {
    is_c++ "$1" && staged_grep -q "#include .*\.cpp" "$1"
}
prevent_cpp_includes_report() {
    echo "Found cpp files included in cpp, hpp. or tpp files."
    pretty_grep "#include .*\.cpp" "$@"
}
prevent_cpp_includes_test() {
    test_check pass foo.hpp ''
    test_check pass foo.tpp ''
    test_check fail foo.hpp '#include "blah/blue/bla.cpp"'
    test_check fail foo.tpp '#include "blah/blue/bla.cpp"'
    test_check fail foo.cpp '#include "blah/blue/bla.cpp"'
    test_check pass foo.hpp '#include "blah/blue/bla.hpp"'
    test_check pass foo.tpp '#include "blah/blue/bla.hpp"'
    test_check pass foo.cpp '#include "blah/blue/bla.hpp"'
    test_check pass foo.hpp '#include "blah/blue/bla.tpp"'
    test_check pass foo.tpp '#include "blah/blue/bla.tpp"'
    test_check pass foo.cpp '#include "blah/blue/bla.tpp"'
    test_check pass foo.hpp 'include "blah/blue/bla.cpp"'
}
standard_checks+=(prevent_cpp_includes)

# Check for Doxygen-comments in cpp files. We don't parse cpp files with
# Doxygen, so they shouldn't contain Doxygen-formatting.
prevent_cpp_doxygen() {
    [[ $1 =~ \.cpp$ ]] && staged_grep -q '/// \|/\*!' "$1"
}
prevent_cpp_doxygen_report() {
    echo "Found Doxygen-formatting in cpp file:"
    pretty_grep '/// \|/\*!' "$@"
    echo "Doxygen-formatting only has an effect in header files."
    echo "Use standard C++ comments in cpp files."
}
prevent_cpp_doxygen_test() {
    test_check fail foo.cpp '/// Dox.'$'\n'
    test_check fail foo.cpp '/*! Dox.'$'\n'
    test_check fail foo.cpp '/*!'$'\n'' * Dox.'$'\n'
    test_check pass foo.cpp '// Dox.'$'\n'
    test_check pass foo.cpp '/* Dox.'$'\n'
    test_check pass foo.cpp '/*'$'\n'' * Dox.'$'\n'
}
standard_checks+=(prevent_cpp_doxygen)

# Check for CMakeLists.txt hardcoding library names
cmakelists_hardcoded_libraries_command_list() {
    printf '%s' add_dependencies
    printf '|%s' add_test_library spectre_python_add_dependencies \
           spectre_python_link_libraries spectre_target_headers \
           spectre_target_sources target_link_libraries target_link_options
}
cmakelists_hardcoded_libraries() {
    local CHECKED_COMMANDS=$(cmakelists_hardcoded_libraries_command_list)
    [[ $1 =~ CMakeLists\.txt$ ]] && \
        whitelist "$1" \
                  "tests/Unit/Executables/CMakeLists.txt$" \
                  "tests/Unit/Parallel/CMakeLists.txt$" \
                  "tests/Unit/ParallelAlgorithms/LinearSolver/\
Actions/CMakeLists.txt$" \
                  "tests/Unit/ParallelAlgorithms/LinearSolver/\
ConjugateGradient/CMakeLists.txt$" \
                  "tests/Unit/ParallelAlgorithms/LinearSolver/Gmres/\
CMakeLists.txt$" \
                  "tests/Unit/ParallelAlgorithms/LinearSolver/Multigrid/\
CMakeLists.txt$" \
                  "tests/Unit/ParallelAlgorithms/LinearSolver/Richardson/\
CMakeLists.txt$" \
                  "tests/Unit/ParallelAlgorithms/LinearSolver/Schwarz/\
CMakeLists.txt$" \
                  "tests/Unit/ParallelAlgorithms/NonlinearSolver/NewtonRaphson/\
CMakeLists.txt$" && \
        staged_grep -E -A1 "(${CHECKED_COMMANDS})\(\$" "$1" | \
            grep -Ev -- "--|${CHECKED_COMMANDS}" | \
            grep -qvF '${'
}
cmakelists_hardcoded_libraries_report() {
    local CHECKED_COMMANDS=$(cmakelists_hardcoded_libraries_command_list)
    echo "Found CMakeLists.txt files using hardcoded library names."
    echo "Use a CMake variable to ensure the library name is consistent" \
         "within the file."
    pretty_grep -E -A1 "(${CHECKED_COMMANDS})\(\$" "$@" | \
        grep -Ev -- "--|${CHECKED_COMMANDS}" | \
        grep -vF '${'
}
cmakelists_hardcoded_libraries_test() {
    test_check pass foo.hpp 'add_dependencies('$'\n''  Name'$'\n'
    test_check fail CMakeLists.txt 'add_dependencies('$'\n''  Name'$'\n'
    test_check pass CMakeLists.txt 'add_dependencies('$'\n''  ${LIBRARY}'$'\n'
    test_check pass foo.hpp 'spectre_target_sources('$'\n''  Name'$'\n'
    test_check fail CMakeLists.txt 'spectre_target_sources('$'\n''  Name'$'\n'
    test_check pass CMakeLists.txt \
               'spectre_target_sources('$'\n''  ${LIBRARY}'$'\n'
    test_check pass foo.hpp 'target_link_options('$'\n''  Name'$'\n'
    test_check fail CMakeLists.txt 'target_link_options('$'\n''  Name'$'\n'
    test_check pass CMakeLists.txt \
               'target_link_options('$'\n''  ${LIBRARY}'$'\n'
}
standard_checks+=(cmakelists_hardcoded_libraries)

# Check for {CHECK,REQUIRE}_THROWS
check_throws() {
    is_c++ "$1" && staged_grep -q -E "(CHECK|REQUIRE)_THROWS[^_]" "$1"
}
check_throws_report() {
    echo "Found use of {CHECK,REQUIRE}_THROWS."
    echo "Use one of the more specific macros, such as CHECK_THROWS_WITH."
    pretty_grep -E "(CHECK|REQUIRE)_THROWS[^_]" "$@"
}
check_throws_test() {
    test_check pass foo.hpp 'CHECK'
    test_check fail foo.hpp 'CHECK_THROWS(foo)'
    test_check pass foo.hpp 'CHECK_THROWS_WITH(foo, bar)'
    test_check pass foo.txt 'CHECK_THROWS(foo)'
    test_check pass foo.hpp 'REQUIRE'
    test_check fail foo.hpp 'REQUIRE_THROWS(foo)'
    test_check pass foo.hpp 'REQUIRE_THROWS_WITH(foo, bar)'
    test_check pass foo.txt 'REQUIRE_THROWS(foo)'
}
standard_checks+=(check_throws)

# Check for typos similar to Doxygen groups
check_dox_groups() {
    is_c++ "$1" && staged_grep -E "^\s*//+\s*[\{\}\(\)]*@[\{\}\(\)]*" "$1" \
            | grep -v -E -q "(^\s*/// @\{|^\s*/// @\})"
}
check_dox_groups_report() {
    echo "Found a likely typo similar to a valid doxygen grouping"
    echo "Use balanced /// @{ and /// @} (on their own lines) to"\
         " create doxygen groups"
    pretty_grep -E "^\s*//+\s*[\{\}\(\)]*@[\{\}\(\)]*" "$1" | \
        grep -v -E "(/// @{|/// @})"
}
check_dox_groups_test() {
    test_check pass foo.hpp '/// @{'
    test_check pass foo.hpp '/// @}'
    test_check pass foo.txt '// @('
    test_check fail foo.hpp '// @{'
    test_check fail foo.hpp '// @}'
    test_check fail foo.hpp '//@{'
    test_check fail foo.hpp '//@}'
    test_check fail foo.hpp '// {@'
    test_check fail foo.hpp '// }@'
    test_check fail foo.hpp '///@('
    test_check fail foo.hpp '///@)'
}
standard_checks+=(check_dox_groups)

# Check for uses of `proxy.ckLocal()`, prefer `Parallel::local(proxy)`
prevent_cklocal() {
    is_c++ "$1" && \
        whitelist "$1" 'src/Parallel/Local.hpp$' && \
        staged_grep -q "\\.ckLocal()" "$1"
}
prevent_cklocal_report() {
    echo "Found occurrences of ckLocal, must use Parallel::local:"
    pretty_grep "\\.ckLocal()" "$@"
}
prevent_cklocal_test() {
    test_check fail foo.cpp 'auto a = some_proxy.ckLocal();'
    test_check pass foo.cpp 'auto a = some_proxy.ckLocalBranch();'
    test_check pass foo.cpp 'auto a = Parallel::local(some_proxy);'
    test_check pass foo.cpp '// a comment talking about ckLocal'
    test_check pass foo.txt '// a comment talking about ckLocal'
}
standard_checks+=(prevent_cklocal)

# Check for uses of `proxy.ckLocalBranch()`,
# prefer `Parallel::local_branch(proxy)`
prevent_cklocalbranch() {
    is_c++ "$1" && \
        whitelist "$1" 'src/Parallel/Local.hpp$' && \
        staged_grep -q "\\.ckLocalBranch()" "$1"
}
prevent_cklocalbranch_report() {
    echo "Found occurrences of ckLocalBranch, must use Parallel::local_branch:"
    pretty_grep "\\.ckLocalBranch()" "$@"
}
prevent_cklocalbranch_test() {
    test_check fail foo.cpp 'auto a = some_proxy.ckLocalBranch();'
    test_check pass foo.cpp 'auto a = some_proxy.ckLocal();'
    test_check pass foo.cpp 'auto a = Parallel::local_branch(some_proxy);'
    test_check pass foo.cpp '// a comment talking about ckLocalBranch'
    test_check pass foo.txt '// a comment talking about ckLocalBranch'
}
standard_checks+=(prevent_cklocalbranch)

# if test is enabled: redefines staged_grep to run tests on files that
# are not in git
[ "$1" = --test ] && staged_grep() { grep "$@"; } && \
    run_tests "${standard_checks[@]}"

# True result for sourcing
:
