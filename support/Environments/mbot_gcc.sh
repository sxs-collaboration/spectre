#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    echo "All modules on Mbot are provided by the system"
}

spectre_load_modules() {
    # The order here is important
    module load gcc/11.4.0
    module load spectre-deps > /dev/null 2>&1
}

spectre_unload_modules() {
    # The order here is important
    module unload spectre-deps > /dev/null 2>&1
    module unload gcc/11.4.0
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules > /dev/null 2>&1
    cmake -D CMAKE_C_COMPILER=gcc \
          -D CMAKE_CXX_COMPILER=g++ \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D MACHINE=Mbot \
          -D USE_XSIMD=yes \
          "$@" \
          $SPECTRE_HOME
}
