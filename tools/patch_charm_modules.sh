#!/usr/bin/env -S bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Applies any patches that are present in $3/tools/CharmModulePatches/ to
# their corresponding .def.h and decl.h files in the charm module outputs.
# This is used to override default charm behavior with custom versions or to
# patch bugs

if [ "$#" -ne 4 ]; then
    echo "Usage: patch_charm_modules.sh module_name path_to_current_source_dir\
 path_to_source_root_dir path_to_current_build_dir"
    exit 1
fi

PATCH_PREFIX=$3/tools/CharmModulePatches${2#"$3"}
DEF_PATCH_FILENAME="$PATCH_PREFIX"/$1.def.h.patch
DECL_PATCH_FILENAME="$PATCH_PREFIX"/$1.decl.h.patch

if [ -f "$DEF_PATCH_FILENAME" ]; then
    patch -u $4/$1.def.h -i $DEF_PATCH_FILENAME
fi

if [ -f "$DECL_PATCH_FILENAME" ]; then
    patch -u $4/$1.decl.h -i $DECL_PATCH_FILENAME
fi
