#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

"$@"

if @WRAP_LIBRARY_LINKER_USE_STUB_OBJECT_FILES@; then
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
