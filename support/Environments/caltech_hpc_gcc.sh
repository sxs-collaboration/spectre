#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    module use /central/groups/sxs/modules/
    echo "Place the following line in your '~/.bashrc' so you don't have to "
    echo "run 'spectre_setup_modules' every time you log in:"
    echo ""
    echo "module use /central/groups/sxs/modules/"
}

spectre_load_modules() {
    module use /central/groups/sxs/modules/
    module load libraries/spectre-deps/skylake-2024-04
}

spectre_unload_modules() {
    module use /central/groups/sxs/modules/
    module unload libraries/spectre-deps/skylake-2024-04
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules

    # Notes:
    # - Choosing the memory allocator to be JEMALLOC is important. When using
    #   the SYSTEM allocator, during BBH runs dumping volume data too often
    #   would result in memory corruption errors. Switching to JEMALLOC fixed
    #   these issues. It is still unclear *why* the SYSTEM allocator caused
    #   these issues. --Kyle Nelli
    # - We turn of docs because we aren't loading a Doxygen module. Could be
    #   added though.
    # - We override the architecture to skylake because that's the oldest type
    #   of nodes on CaltechHPC. All dependencies were also compiled for skylake.
    #   This means we should be able to run on all types of nodes available on
    #   the cluster.
    # - We temporarily turn off ParaView until a compatible version is
    #   installed.
    cmake -D CMAKE_C_COMPILER=gcc \
          -D CMAKE_CXX_COMPILER=g++ \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D CHARM_ROOT=$CHARM_ROOT \
          -D BLA_VENDOR=OpenBLAS \
          -D CMAKE_BUILD_TYPE=Release \
          -D BUILD_DOCS=OFF \
          -D DEBUG_SYMBOLS=OFF \
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D MACHINE=CaltechHpc \
          -D OVERRIDE_ARCH=skylake \
          -D ENABLE_PARAVIEW=OFF \
          "$@" \
          $SPECTRE_HOME
}
