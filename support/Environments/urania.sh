#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_charm_paths() {
    # Define Charm paths
    export CHARM_ROOT=/u/guilara/charm_impi_2/mpi-linux-x86_64-smp
    export PATH=$PATH:/u/guilara/charm_impi_2/mpi-linux-x86_64-smp/bin
}

spectre_load_modules() {
    module load gcc/11
    module load impi/2021.7
    module load boost/1.79
    module load gsl/1.16
    module load cmake/3.26
    module load hdf5-serial/1.12.2
    module load anaconda/3/2021.11
    module load paraview/5.10
    # Load Spack environment
    source /u/guilara/repos/spack/share/spack/setup-env.sh
    source /urania/u/guilara/repos/spack/var/spack/environments\
/env3_spectre_impi/loads
    # Load python environment
    source /u/guilara/envs/spectre_env
}

spectre_unload_modules() {
    module unload gcc/11
    module unload impi/2021.7
    module unload boost/1.79
    module unload gsl/1.16
    module unload cmake/3.26
    module unload hdf5-serial/1.12.2
    module unload anaconda/3/2021.11
    module unload paraview/5.10
    # Unload Spack environment
    spack env deactivate
    # Unload python environment
    deactivate
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
          -D CMAKE_BUILD_TYPE=Release \
          -D DEBUG_SYMBOLS=OFF \
          -D BUILD_SHARED_LIBS=ON \
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D MACHINE=Urania \
          -D Catch2_DIR=/u/guilara/repos/Catch2/install_dir/lib64/cmake/Catch2
          -D MPI_C_COMPILER=/mpcdf/soft/SLE_15/packages/skylake\
/impi/gcc_11-11.2.0/2021.7.1/bin/mpigcc \
          -D MPI_CXX_COMPILER=/mpcdf/soft/SLE_15/packages/skylake\
/impi/gcc_11-11.2.0/2021.7.1/bin/mpig++ \
          -D MPI_Fortran_COMPILER=/mpcdf/soft/SLE_15/packages/skylake\
/impi/gcc_11-11.2.0/2021.7.1/bin/mpigfortran \
          -Wno-dev "$@" $SPECTRE_HOME
}
