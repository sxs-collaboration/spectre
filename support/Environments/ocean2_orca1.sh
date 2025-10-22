#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    echo "All modules on Ocean2 are provided by the system"
}

spectre_unload_modules() {
    module unload gnu12/12.3.0
    module unload intel/mpi/2021.16
    module unload git/2.43.0
    module unload cmake/3.24.2
    module unload python/3.12.4
    module unload orca1/openblas/0.3.27
    module unload orca1/boost/1.85.0
    module unload orca1/gsl/2.8
    module unload orca1/hdf5/1.12.3
    module unload orca1/charm/8.0.0
    module unload orca1/libxsmm/1.16.1
    module unload orca1/catch2/3.5.4
    module unload orca1/yaml-cpp/0.7.0
    module unload blaze/3.8
    module unload xsimd/13.2.0
}

spectre_load_modules() {
    module load gnu12/12.3.0
    module load intel/mpi/2021.16
    module load git/2.43.0
    module load cmake/3.24.2
    module load python/3.12.4
    module load orca1/openblas/0.3.27
    module load orca1/boost/1.85.0
    module load orca1/gsl/2.8
    module load orca1/hdf5/1.12.3
    module load orca1/charm/8.0.0
    module load orca1/libxsmm/1.16.1
    module load orca1/catch2/3.5.4
    module load orca1/yaml-cpp/0.7.0
    module load blaze/3.8
    module load xsimd/13.2.0
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake -D CMAKE_C_COMPILER=gcc \
          -D CMAKE_CXX_COMPILER=g++ \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D CHARM_ROOT=$CHARM_ROOT \
          -D BLA_VENDOR=OpenBLAS \
          -D CMAKE_BUILD_TYPE=Release \
          -D BUILD_DOCS=OFF \
          -D DEBUG_SYMBOLS=OFF \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D SPECTRE_FETCH_MISSING_DEPS=OFF \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D BUILD_SHARED_LIBS=ON \
          -D MACHINE=Ocean2_orca1 \
          -D ENABLE_PARAVIEW=OFF \
          -D BOOTSTRAP_PY_DEPS=ON \
          "$@" \
          $SPECTRE_HOME
}
