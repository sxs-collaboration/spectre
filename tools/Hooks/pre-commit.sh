#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

###############################################################################
# Add grep colors if available
color_option=''
if grep --help 2>&1 | grep -q -e --color
then
  color_option='--color=auto'
fi

###############################################################################
# Get list of non-deleted file names, which we need below.
commit_files=""

for var in `@GIT_EXECUTABLE@ diff --cached --name-only`
do
    if [ -f $var ]; then
        commit_files="${commit_files} $var"
    fi
done

# If no files were added or modified we do not run any of the hooks. If we did
# then rewording commits, and making empty commits would result in the hooks
# searching the entire repository, which is not what we want in general.
if [ -z "$commit_files" ]; then
    exit 0
fi

###############################################################################
# Check the file size
@PYTHON_EXECUTABLE@ @CMAKE_SOURCE_DIR@/.git/hooks/CheckFileSize.py
ret_code=$?
if [ "$ret_code" -ne 0 ]; then
    @GIT_EXECUTABLE@ stash pop > /dev/null 2>&1
    exit 1
fi

###############################################################################
# Check for lines longer than 80 characters
found_long_lines=`find $commit_files -name '*.[ch]pp' \
| xargs grep --with-filename -n '.\{81,\}'`
if [[ $found_long_lines != "" ]]; then
    echo "Found lines over 80 characters:"
    echo "$found_long_lines"
    exit 1
fi

###############################################################################
# Find lines that have tabs in them and block them from being committed
if @GIT_EXECUTABLE@ diff --cached | grep -E '^\+.*'$'\t' > /dev/null
then
cat<<END;
error: tabulations were found in staged changes.

If you *really* know what you are doing, you can force this commit
with 'git commit --no-verify'. However, this is *very* strongly
discouraged. Unless the language of the file requires tabs
(e.g. Makefiles) please switch to spaces.

Incriminating changes:
END
  for i in `@GIT_EXECUTABLE@ diff --cached --name-only`
  do
    @GIT_EXECUTABLE@ blame $i \
  | sed 's@[0-9a-f]\{1,\} (Not Committed Yet [^\)]\+ \{1,\}\([0-9]\{1,\}\)) \(.*\)@'$i':\1: \2@g;tx;d;:x' \
  | GREP_COLOR='1;37;41' grep -F $'\t' $color_option
  done
  exit 1
fi

###############################################################################
# Find files that have white space at the end of a line and block them
if @GIT_EXECUTABLE@ diff --cached | grep -E '^\+.* +$' > /dev/null
then
cat<<END;
error: spaces at end of lines were found in staged changes.

Incriminating changes:
END
  for i in `@GIT_EXECUTABLE@ diff --cached --name-only`
  do
    @GIT_EXECUTABLE@ blame $i \
  | sed 's@[0-9a-f]\{1,\} (Not Committed Yet [^\)]\+ \{1,\}\([0-9]\{1,\}\)) \(.*\)@'$i':\1: \2@g;tx;d;:x' \
  | grep -v -E '^[^:]*:[^:]+: $' | GREP_COLOR='1;37;41' grep -E ' +$' $color_option
  done
  exit 1
fi

###############################################################################
# Make sure all files have the license header in them.
no_license_found=`grep -L "^.*Distributed under the MIT License" $commit_files`
if [[ $no_license_found != "" ]]; then
    echo "Did not find the license header in these files:"
    echo "$no_license_found"
    exit 1
fi

###############################################################################
# Check to make sure all files have a newline at the end
no_newline=
for i in `@GIT_EXECUTABLE@ diff --cached --name-only`
do
  if [ -e "$i" ] && [ $(tail -n 1 "$i" | wc -l | tr -d ' ') != 1 ]
  then
    no_newline=$no_newline$i$'\n'
  fi
done
if [ "$no_newline" != "" ]
then
cat<<END;
error: no newline at end of file

Incriminating files:
END
  echo $no_newline
  exit 1
fi

###############################################################################
if @GIT_EXECUTABLE@ diff --cached | grep -E '^\+.*'$'\r' > /dev/null
then
cat<<END;
error: carriage returns were found in staged changes.

Incriminating files:
END
  for i in `@GIT_EXECUTABLE@ diff --cached --name-only`
  do
    @GIT_EXECUTABLE@ blame $i \
  | sed 's@[0-9a-f]\{1,\} (Not Committed Yet [^\)]\+ \{1,\}\([0-9]\{1,\}\)) \(.*\)@'$i':\1: \2@g;tx;d;:x' \
  | grep -v -E '^[^:]*:[^:]+: $' | GREP_COLOR='1;37;41' grep -E '\r+$' $color_option
  done
  exit 1
fi

###############################################################################
# Check for tests using Catch's TEST_CASE instead of SPECTRE_TEST_CASE
found_test_case=`
find $commit_files -name '*.[ch]pp' \
    | xargs grep --with-filename -n "^TEST_CASE"`
if [[ $found_test_case != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    echo "Found occurrences of TEST_CASE, must use SPECTRE_TEST_CASE:"
    echo "$found_test_case"
    exit 1
fi

###############################################################################
# Check for tests using Catch's Approx, which has a very loose tolerance
found_bad_approx=`
find $commit_files -name '*.[ch]pp' \
    | xargs grep --with-filename -n "Approx("`
if [[ $found_bad_approx != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    printf "Found occurrences of Approx, must use approx from " \
           "SPECTRE_ROOT/tests/Unit/TestHelpers.hpp instead:\n"
    echo "$found_bad_approx"
    exit 1
fi

###############################################################################
# Check for Doxygen comments on the same line as a /*!
found_bad_doxygen_syntax=`
find $commit_files -type f -name '*.[ch]pp' \
    | xargs grep --with-filename -n '/\*\! '`
if [[ $found_bad_doxygen_syntax != "" ]]; then
    echo "This script can be run locally from any source dir using:"
    echo "SPECTRE_ROOT/tools/CheckFiles.sh"
    printf "Found occurrences of bad Doxygen syntax: /*! STUFF\n"
    echo $found_bad_doxygen_syntax | \
        GREP_COLOR='1;37;41' grep -E '\/\*\!.*' $color_option
    echo ''
    exit 1
fi

###############################################################################
# Use git-clang-format to check for any suspicious formatting of code.
@PYTHON_EXECUTABLE@ @CMAKE_SOURCE_DIR@/.git/hooks/ClangFormat.py
