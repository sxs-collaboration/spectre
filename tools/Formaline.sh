#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Generate an archive of the source tree, then generate an object file to
# be linked into the final executable.
formaline_archive_name=spectre_$1
formaline_dir=@CMAKE_BINARY_DIR@/tmp/
pushd @CMAKE_SOURCE_DIR@ >/dev/null
git ls-tree --full-tree --name-only HEAD \
    | xargs tar -czf ${formaline_dir}/${formaline_archive_name}.tar.gz
popd >/dev/null

pushd @CMAKE_BINARY_DIR@/tmp >/dev/null
ld -r -b binary -o ${formaline_archive_name}.o ${formaline_archive_name}.tar.gz
rm ${formaline_dir}/${formaline_archive_name}.tar.gz
popd >/dev/null
# Set the formaline object file output name
formaline_object_output=${formaline_dir}/${formaline_archive_name}.o
# formaline takes the archive name and the name of the output file
formaline_output=@CMAKE_BINARY_DIR@/tmp/${formaline_archive_name}.cpp
rm -f ${formaline_output}
cat >${formaline_output} <<EOF
#include <string>
#include <vector>

extern char _binary_${formaline_archive_name}_tar_gz_start[];
extern char _binary_${formaline_archive_name}_tar_gz_end[];

namespace formaline {
std::vector<char> get_archive() noexcept {
  return std::vector<char>{_binary_${formaline_archive_name}_tar_gz_start,
                           _binary_${formaline_archive_name}_tar_gz_end};
}

std::string get_environment_variables() noexcept {
  // Use a delimiter that's unlikely to appear in the printenv
  // output.
  return R"AZBYCXDWEVFU(
`printenv`
)AZBYCXDWEVFU";
}

std::string get_library_versions() noexcept {
  return R"AZBYCXDWEVFU(
`cat @CMAKE_BINARY_DIR@/LibraryVersions.txt`
)AZBYCXDWEVFU";
}

std::string get_paths() noexcept {
  return R"AZBYCXDWEVFU(PATH=${PATH}
CPATH=${CPATH}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
LIBRARY_PATH=${LIBRARY_PATH}
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
)AZBYCXDWEVFU";
}
}  // namespace formaline
EOF
