#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Environment for the ocean cluster, located at Cal State Fullerton.
#
# Access ocean via `ssh ocean.fullerton.edu`
#
# For questions regarding ocean or to request access, please contact
# Geoffrey Lovelace by email (glovelace at fullerton dot edu).
# Access is normally restricted to members of Cal State Fullerton's
# Nicholas and Lee Begovich Center for Gravitational-Wave Physics and Astronomy
# and their collaborators.

spectre_setup_modules() {
    echo "All modules on ocean are provided by the system"
}

spectre_unload_modules() {
    module unload prun/1.2
    module unload gmp/4.3.2
    module unload mpc/1.0.3
    module unload mpf4/3.1.6
    module unload gnu11/11.3.0
    module unload openmpi/4.1.4
    module unload llvm/13.0.1
    module unload cmake/3.24.1
    module unload python/3.9.5
    module unload openblas-0.3.20-gcc-11.3.0-tc4qxfv
    module unload zlib-1.2.12-gcc-11.3.0-ge3ye5j
    module unload blaze-3.8-gcc-11.3.0-y7sgzzc
    module unload brigand-master-gcc-11.3.0-nsmmsxm
    module unload libsharp-1.0.0-gcc-11.3.0-w7e7n5z
    module unload catch2/3.4.0
    module unload gsl-2.7.1-gcc-11.3.0-sey3z3o
    module unload jemalloc-5.2.1-gcc-11.3.0-r63hemp
    module unload yaml-cpp-0.7.0-gcc-11.3.0-a4rumor
    module unload boost-1.79.0-gcc-11.3.0-ck2cccn
    module unload hdf5-1.12.2-gcc-11.3.0-dly2yyu
    module unload binutils-2.38-gcc-11.3.0-fmchbp7
    module unload libxsmm/1.16.1
    module unload charm-7.0.0-gnu11-clang-smp-patch
    module unload git/2.19.6
    module unload doxygen/1.9.5
    module unload fftw-3.3.10-gcc-11.3.0-g2odatg
}

spectre_load_modules() {
    module purge
    module load prun/1.2
    module load gmp/4.3.2
    module load mpc/1.0.3
    module load mpfr/3.1.6
    module load gnu11/11.3.0
    module load openmpi/4.1.4
    module load llvm/13.0.1
    module load cmake/3.24.1
    module load python/3.9.5
    export MODULEPATH=$MODULEPATH:/opt/ohpc/pub/apps\
/spack2022/share/spack/modules/linux-centos7-broadwell/
    module load openblas-0.3.20-gcc-11.3.0-tc4qxfv
    module load zlib-1.2.12-gcc-11.3.0-ge3ye5j
    module load blaze-3.8-gcc-11.3.0-y7sgzzc
    module load brigand-master-gcc-11.3.0-nsmmsxm
    module load libsharp-1.0.0-gcc-11.3.0-w7e7n5z
    module load catch2/3.4.0
    module load gsl-2.7.1-gcc-11.3.0-sey3z3o
    module load jemalloc-5.2.1-gcc-11.3.0-r63hemp
    module load yaml-cpp-0.7.0-gcc-11.3.0-a4rumor
    module load boost-1.79.0-gcc-11.3.0-ck2cccn
    module load hdf5-1.12.2-gcc-11.3.0-dly2yyu
    module load binutils-2.38-gcc-11.3.0-fmchbp7
    module load libxsmm/1.16.1
    module load charm-7.0.0-gnu11-clang-smp-patch
    module load git/2.19.6
    module load doxygen/1.9.5
    module load fftw-3.3.10-gcc-11.3.0-g2odatg
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    export GCC_HOME=/opt/ohpc/pub/compiler/gcc/11.3.0/bin
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_C_COMPILER=gcc \
          -D CMAKE_CXX_COMPILER=g++ \
          -D CMAKE_Fortran_COMPILER=${GCC_HOME}/gfortran \
          -D USE_PCH=ON \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D BOOTSTRAP_PY_DEPS=ON \
          "$@" \
          $SPECTRE_HOME
}
