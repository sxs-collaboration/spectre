#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

pushd @CMAKE_SOURCE_DIR@ >/dev/null
git_commit_hash=`@GIT_EXECUTABLE@ describe --abbrev=0 --always --tags`
git_branch=`@GIT_EXECUTABLE@ rev-parse --abbrev-ref HEAD`
popd >/dev/null

# Create a copy of InfoFromBuild.cpp based on the output filename.
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
InfoFromBuild_file=@CMAKE_BINARY_DIR@/tmp/\
$(basename "${!oindex}")_InfoFromBuild.cpp
cp @CMAKE_BINARY_DIR@/Informer/InfoFromBuild.cpp "${InfoFromBuild_file}"

# Formaline through the linker doesn't work on macOS and since we won't
# be doing production runs on macOS we disable it.
if [ -f @CMAKE_BINARY_DIR@/tmp/Formaline.sh ]; then
    . @CMAKE_BINARY_DIR@/tmp/Formaline.sh $(basename "${!oindex}")
    "$@" -DGIT_COMMIT_HASH=$git_commit_hash -DGIT_BRANCH=$git_branch \
         "${InfoFromBuild_file}" "${formaline_output}" \
         ${formaline_object_output}
    rm ${formaline_output}
    rm ${formaline_object_output}
else
    "$@" -DGIT_COMMIT_HASH=$git_commit_hash -DGIT_BRANCH=$git_branch \
         "${InfoFromBuild_file}"
fi

rm ${InfoFromBuild_file}
