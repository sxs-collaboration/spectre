#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    echo "All modules on Deucalion are provided by the system"
}


spectre_load_modules() {
    module load Boost/1.88.0-GCC-14.3.0
    module load foss/2026.1
    module load HDF5/2.1.1-gompi-2026.1
    module load GSL/2.8-GCC-15.2.0
    module load git/2.52.0
    module load CMake/4.2.1-GCCcore-15.2.0
    module load Python/3.14.2-GCCcore-15.2.0
    module load binutils/2.45
}

spectre_unload_modules() {
    module unload Boost/1.88.0-GCC-14.3.0
    module unload foss/2026.1
    module unload HDF5/2.1.1-gompi-2026.1
    module unload GSL/2.8-GCC-15.2.0
    module unload git/2.52.0
    module unload CMake/4.2.1-GCCcore-15.2.0
    module unload Python/3.14.2-GCCcore-15.2.0
    module unload binutils/2.45
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    if [ -z ${CHARM_ROOT} ]; then
        echo "You must set CHARM_ROOT to the charm directory you want"
        return 1
    fi
    spectre_load_modules

    # Notes:
    # - We turn off docs because we aren't loading a Doxygen module. Could be
    #   added though.
    # export CHARM_ROOT /path/to/charm
    cmake -D CMAKE_CXX_COMPILER=g++ \
          -D CMAKE_C_COMPILER=gcc \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D USE_LD=ld \
          -D CHARM_ROOT=$CHARM_ROOT \
          -D ENABLE_PYTHON=ON \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D CMAKE_BUILD_TYPE=Release \
          -D SPECTRE_FETCH_MISSING_DEPS=ON \
          -D BOOTSTRAP_PY_DEPS=ON \
          -D BUILD_TESTING=OFF \
          -D BUILD_DOCS=OFF \
          -D DEBUG_SYMBOLS=OFF \
          #-D USE_CCACHE=ON \
          -D MACHINE=Deucalion \
          "$@" \
          $SPECTRE_HOME
}
