#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    echo "All modules on Sonic are provided by the system"
}

spectre_load_modules() {
    # The order here is important
    module load sxs
    module load spectre-env > /dev/null 2>&1
}

spectre_unload_modules() {
    # The order here is important
    module unload spectre-env > /dev/null 2>&1
    module unload sxs
}

spectre_run_cmake_clang() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    spectre_load_modules > /dev/null 2>&1
    cmake -D CMAKE_C_COMPILER=clang \
          -D CMAKE_CXX_COMPILER=clang++ \
          -D CMAKE_Fortran_COMPILER=flang \
          -D CHARM_ROOT=$CHARM_ROOT \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D CMAKE_BUILD_TYPE=Release \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D ENABLE_PARAVIEW=ON \
          -D MACHINE=Sonic \
          -D USE_XSIMD=yes \
          -D DEBUG_SYMBOLS=OFF \
          "$@" \
          $SPECTRE_HOME
}

spectre_run_cmake_gcc() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    spectre_load_modules > /dev/null 2>&1
    cmake -D CMAKE_C_COMPILER=gcc \
          -D CMAKE_CXX_COMPILER=g++ \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D CHARM_ROOT=$CHARM_ROOT \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D CMAKE_BUILD_TYPE=Release \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D ENABLE_PARAVIEW=ON \
          -D MACHINE=Sonic \
          -D USE_XSIMD=yes \
          -D DEBUG_SYMBOLS=OFF \
          "$@" \
          $SPECTRE_HOME
}
