#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    module use /central/groups/sxs/modules
    echo "Place the following line in you '~/.bashrc' so you don't have to "
    echo "run 'spectre_setup_modules' every time you log in:"
    echo ""
    echo "module use /central/groups/sxs/modules"
}

spectre_load_modules() {
    module use /central/groups/sxs/modules
    module load libraries/spectre-deps/icelake-2024-02 > /dev/null 2>&1
}

spectre_unload_modules() {
    module use /central/groups/sxs/modules
    module unload libraries/spectre-deps/icelake-2024-02 > /dev/null 2>&1
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules > /dev/null 2>&1

    # Note that choosing the memory allocator to be JEMALLOC is important. When
    # using the SYSTEM allocator, during BBH runs dumping volume data too often
    # would result in memory corruption errors. Switching to JEMALLOC fixed
    # these issues. It is still unclear *why* the SYSTEM allocator caused these
    # issues. --Kyle Nelli
    cmake -D CMAKE_C_COMPILER=gcc \
          -D CMAKE_CXX_COMPILER=g++ \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D MACHINE=CaltechHpcIcelake \
          -D OVERRIDE_ARCH=icelake-server \
          -D BLA_VENDOR=OpenBLAS \
          "$@" \
          $SPECTRE_HOME
}
