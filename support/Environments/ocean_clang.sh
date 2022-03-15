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
# Gravitational-Wave Physics and Astronomy Center and their collaborators.

spectre_setup_modules() {
    echo "All modules on ocean are provided by the system"
}

spectre_unload_modules() {
    module unload ohpc
    module unload llvm/10.0.1
    module unload gnu7/7.3.0
    module unload openmpi/1.10.7
    module unload prun/1.2
    module unload cmake/3.18.5
    module unload git-2.19.2-gcc-7.3.0-jfnpgdh
    module unload blaze/3.8
    module unload brigand-master-gcc-7.3.0-gwg63zg
    module unload libsharp-2018-01-17-gcc-7.3.0-4xamgaw
    module unload catch2-2.11.3-gcc-7.3.0-l7lqzrg
    module unload gsl-2.5-gcc-7.3.0-i7icadp
    module unload jemalloc-4.5.0-gcc-7.3.0-wlf2m7r
    module unload libxsmm/1.16.1
    module unload yaml-cpp-develop-gcc-7.3.0-qcfbbll
    module unload boost-1.68.0-gcc-7.3.0-vgl6ofr
    module unload hdf5-1.12.0-gcc-7.3.0-mknp6xv
    module unload openblas-0.3.4-gcc-7.3.0-tt2coe7
    module unload python/3.9.5
    module unload charm-6.10.2-libs
    module unload doxygen-1.9.1-gcc-7.3.0-nxmwu4a
    module unload zlib-1.2.11-gcc-7.3.0-h3h2oa4
}

spectre_load_modules() {
    module load ohpc
    module load python/3.9.5
    module load gnu7/7.3.0
    module load openmpi/1.10.7
    module load prun/1.2
    module load llvm/10.0.1
    source /opt/ohpc/pub/apps/spack/0.12.0/share/spack/setup-env.sh
    module load cmake/3.18.5
    module load git-2.19.2-gcc-7.3.0-jfnpgdh
    module load blaze/3.8
    module load brigand-master-gcc-7.3.0-gwg63zg
    module load libsharp-2018-01-17-gcc-7.3.0-4xamgaw
    module load catch2-2.11.3-gcc-7.3.0-l7lqzrg
    module load gsl-2.5-gcc-7.3.0-i7icadp
    module load jemalloc-4.5.0-gcc-7.3.0-wlf2m7r
    module load libxsmm/1.16.1
    module load yaml-cpp-develop-gcc-7.3.0-qcfbbll
    module load boost-1.68.0-gcc-7.3.0-vgl6ofr
    module load hdf5-1.12.0-gcc-7.3.0-mknp6xv
    module load openblas-0.3.4-gcc-7.3.0-tt2coe7
    module load charm-6.10.2-libs
    module load doxygen-1.9.1-gcc-7.3.0-nxmwu4a
    module load zlib-1.2.11-gcc-7.3.0-h3h2oa4
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    export GCC_HOME=/opt/ohpc/pub/compiler/gcc/7.3.0/bin
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_C_COMPILER=clang \
          -D CMAKE_CXX_COMPILER=clang++ \
          -D CMAKE_Fortran_COMPILER=${GCC_HOME}/gfortran \
          "$@" \
          $SPECTRE_HOME
}
