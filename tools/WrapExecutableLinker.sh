#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

temp_files=()
trap 'rm -r "${temp_files[@]}"' EXIT

pushd @CMAKE_SOURCE_DIR@ >/dev/null
git_description="@GIT_DESCRIPTION@"
git_branch="@GIT_BRANCH@"
if [ -z "$git_description" ]; then
    git_description="NOT_IN_GIT_REPO"
fi
if [ -z "$git_branch" ]; then
    git_branch="NOT_IN_GIT_REPO"
fi
popd >/dev/null

# Create a copy of InfoAtLink.cpp based on the output filename.
# When compiling an executable from a cpp file, charmc writes a
# temporary file called ${basename}.o to the current directory, so we
# need all the compilations that might run in parallel in one
# directory to have different base names for cpp files.
oindex=
# Find the index of "-o" in the argument list
for (( i=1 ; i<$# ; ++i )) ; do
    [ "${!i}" = -o ] && { oindex=$i ; break ; }
done
if [ -z "${oindex}" ] ; then
    echo "$0: Can't find -o option" >&2
    exit 1
fi
# Construct a filename from the next argument
let ++oindex
InfoAtLink_file=@CMAKE_BINARY_DIR@/tmp/\
$(basename "${!oindex}")_InfoAtLink.cpp
temp_files+=("${InfoAtLink_file}")
cp @CMAKE_BINARY_DIR@/Informer/InfoAtLink.cpp "${InfoAtLink_file}"
# Read the appropriate flags for compiling InfoAtLink.cpp from the generated
# file
InfoAtLink_flags=`cat @CMAKE_BINARY_DIR@/Informer/InfoAtLink_flags.txt`

# - Formaline through the linker doesn't work on macOS and since we won't
#   be doing production runs on macOS we disable it.
if [ -f @CMAKE_BINARY_DIR@/tmp/Formaline.sh ]; then
    . @CMAKE_BINARY_DIR@/tmp/Formaline.sh $(basename "${!oindex}")
    temp_files+=("${formaline_output}" "${formaline_object_output}")
    "$@" -DGIT_DESCRIPTION=$git_description -DGIT_BRANCH=$git_branch \
         ${InfoAtLink_flags} "${InfoAtLink_file}" "${formaline_output}" \
         ${formaline_object_output}
else
    "$@" -DGIT_DESCRIPTION=$git_description -DGIT_BRANCH=$git_branch \
         ${InfoAtLink_flags} "${InfoAtLink_file}"
fi

if @WRAP_EXECUTABLE_LINKER_USE_STUB_OBJECT_FILES@; then
    # When linking, mv the object files to a temporary file (this
    # preserves times), then use touch to create a new empty
    # stub object file with the same times (atime and mtime) as
    # the original object file, except that it is 0kB
    for file in "$@"
    do
        if [[ $file =~ .*\.cpp\.o ]]; then
            temp_file="${file}.tmp_to_eliminate_size"
            mv "${file}" "${temp_file}"
            touch -r "${temp_file}" "${file}"
            rm "${temp_file}"
        fi
    done
fi
