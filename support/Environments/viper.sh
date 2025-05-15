#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_load_modules() {
    module load gcc/13
    module load impi/2021.11
    module load boost/1.83
    module load gsl/2.7
    module load cmake/3.30
    module load hdf5-serial/1.14.1
    module load python-waterboa/2024.06
    module load doxygen/1.10.0
    module load paraview/5.11
    # Load Spack environment
    source /u/guilara/repos/spack/share/spack/setup-env.sh
    spack env activate env3_spectre_impi
    # Load python environment
    source /u/guilara/envs/spectre_env/bin/activate
    # Define Charm paths
    export CHARM_ROOT=/u/guilara/charm_impi/mpi-linux-x86_64-smp
    export PATH=$PATH:/u/guilara/charm_impi/mpi-linux-x86_64-smp/bin
}

spectre_unload_modules() {
    module load gcc/13
    module load impi/2021.11
    module load boost/1.83
    module load gsl/2.7
    module load cmake/3.30
    module load hdf5-serial/1.14.1
    module load python-waterboa/2024.06
    module load doxygen/1.10.0
    module load paraview/5.11
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
          -D MACHINE=Viper \
          -D SPEC_ROOT=/u/guilara/repos/spec \
          -D Catch2_DIR=/u/guilara/repos/Catch2/install_dir/lib64/cmake/Catch2 \
          "$@" $SPECTRE_HOME
}
