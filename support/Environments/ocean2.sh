#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    echo "All modules on Ocean2 are provided by the system"
}

spectre_unload_modules() {
    module unload gnu12/12.3.0
    module unload intel/mpi/2021.16
    module unload cmake/3.24.2
    module unload openblas/0.3.27
    module unload blaze/3.8
    module unload boost/1.85.0
    module unload catch2/3.5.4
    module unload gsl/2.8
    module unload hdf5/1.14.6
    module unload jemalloc/5.3.0
    module unload libxsmm/1.17
    module unload yaml-cpp/0.7.0
    module unload libffi/3.4.5
    module unload python/3.12.4
    module unload python/spectre-python-2025.08.19
    module unload llvm/18.1.8
    module unload libbacktrace/2024.07.09
    module unload yasm/1.3.0
    module unload ffmpeg/7.0.1
    module unload fftw/3.3.10
    module unload petsc/3.21.3
    module unload charm/8.0.0
    module unload libbacktrace/2024.07.09
    module unload xsimd/13.2.0
}

spectre_load_modules() {
    module load gnu12/12.3.0
    module load intel/mpi/2021.16
    module load cmake/3.24.2
    module load openblas/0.3.27
    module load blaze/3.8
    module load boost/1.85.0
    module load catch2/3.5.4
    module load gsl/2.8
    module load hdf5/1.14.6
    module load jemalloc/5.3.0
    module load libxsmm/1.17
    module load yaml-cpp/0.7.0
    module load libffi/3.4.5
    module load python/3.12.4
    module load python/spectre-python-2025.08.19
    module load llvm/18.1.8
    module load libbacktrace/2024.07.09
    module load yasm/1.3.0
    module load ffmpeg/7.0.1
    module load fftw/3.3.10
    module load petsc/3.21.3
    module load charm/8.0.0
    module load libbacktrace/2024.07.09
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
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D BUILD_SHARED_LIBS=ON \
          -D MACHINE=Ocean2 \
          -D ENABLE_PARAVIEW=OFF \
          -D BOOTSTRAP_PY_DEPS=ON \
          "$@" \
          $SPECTRE_HOME
}
