#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

temp_files=()
trap 'rm -rf "${temp_files[@]}"' EXIT

pushd @CMAKE_SOURCE_DIR@ >/dev/null
git_description="@GIT_DESCRIPTION_COMMAND@"
git_branch="@GIT_BRANCH_COMMAND@"
if [ ! -z "${git_description}" ]; then
    git_description=`${git_description}`
fi
if [ ! -z "${git_branch}" ]; then
    git_branch=`${git_branch}`
fi
popd >/dev/null

# Find the index of "-o" in the argument list
oindex=
for (( i=1 ; i<$# ; ++i )) ; do
    [ "${!i}" = -o ] && { oindex=$i ; break ; }
done
if [ -z "${oindex}" ] ; then
    echo "$0: Can't find -o option" >&2
    exit 1
fi
# Construct a filename from the next argument
let ++oindex
executable_name=$(basename "${!oindex}")

# We compile the InfoAtLink.cpp file into the executable at link time
InfoAtLink_file=@CMAKE_SOURCE_DIR@/src/Informer/InfoAtLink.cpp
# Read the appropriate flags for compiling InfoAtLink.cpp from the generated
# file
InfoAtLink_flags=`cat @CMAKE_BINARY_DIR@/Informer/InfoAtLink_flags.txt`
# Compile into an object file
InfoAtLink_ofile=@CMAKE_BINARY_DIR@/tmp/${executable_name}_InfoAtLink.cpp.o
@CMAKE_CXX_COMPILER@ -c ${InfoAtLink_file} \
    -DGIT_DESCRIPTION=$git_description -DGIT_BRANCH=$git_branch \
    -DLINK_TIME="$(date -u -Iseconds)" ${InfoAtLink_flags} \
    -DEXECUTABLE_NAME="${executable_name}" \
    -o ${InfoAtLink_ofile}
temp_files+=("${InfoAtLink_ofile}")

# We inject the source code archive into the executable at link time
# by compiling it into an object file (formaline).
# - Formaline through the linker doesn't work on macOS and since we won't
#   be doing production runs on macOS we disable it.
if [ -f @CMAKE_BINARY_DIR@/tmp/Formaline.sh ]; then
    . @CMAKE_BINARY_DIR@/tmp/Formaline.sh ${executable_name}
    formaline_ofile=@CMAKE_BINARY_DIR@/tmp/${executable_name}_Formaline.cpp.o
    @CMAKE_CXX_COMPILER@ -c ${formaline_output} ${InfoAtLink_flags} \
        -o ${formaline_ofile}
    temp_files+=("${formaline_output}" "${formaline_object_output}" \
        "${formaline_ofile}")
    "$@" ${InfoAtLink_ofile} ${formaline_ofile} ${formaline_object_output}
else
    "$@" ${InfoAtLink_ofile}
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
