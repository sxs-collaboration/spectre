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
    # System-installed
    module load gcc/9.2.0
    module load oneapi/mpi/2021.5.1
    module load mkl/18.1
    module load hdf5/1.12.1
    module load cmake/3.18.0
    module load git/2.37.2
    # Installed in SXS group
    module load blaze/3.8
    module load boost/1.82.0
    module load brigand/master
    module load catch/2.13.9
    module load gsl/2.4
    module load libsharp/1.0.0
    module load libxsmm/1.16.1
    module load envs/spectre-python
    module load yaml-cpp/0.6.2
    module load charm/7.0.0
    module load spec-exporter/2023-07
}

spectre_unload_modules() {
    # System-installed
    module unload gcc/9.2.0
    module unload oneapi/mpi/2021.5.1
    module unload mkl/18.1
    module unload hdf5/1.12.1
    module unload cmake/3.18.0
    module unload git/2.37.2
    # Installed in SXS group
    module unload blaze/3.8
    module unload boost/1.82.0
    module unload brigand/master
    module unload catch/2.13.9
    module unload gsl/2.4
    module unload libsharp/1.0.0
    module unload libxsmm/1.16.1
    module unload envs/spectre-python
    module unload yaml-cpp/0.6.2
    module unload charm/7.0.0
    module unload spec-exporter/2023-07
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D MACHINE=CaltechHpc \
          -D OVERRIDE_ARCH=skylake-avx512 \
          "$@" \
          $SPECTRE_HOME
}
