#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Load system modules
spectre_load_sys_modules() {
    # Make sure the core default modules are loaded.
    # Without these you can get weird compilation errors
    # that make no sense.
    module load shared cpu sdsc DefaultModules slurm/expanse numactl

    module load gcc/10.2.0
    module load openblas/0.3.10-openmp
    module load gsl/2.5
    module load boost/1.74.0
    module load cmake/3.18.2
    module load hwloc
    module load libunwind
    module load intel-mpi/2019.8.254
    module load hdf5/1.10.7
    module load python/3.8.5
    module load doxygen/1.8.17
}

# Unload system modules
spectre_unload_sys_modules() {
    module unload doxygen/1.8.17
    module unload python/3.8.5
    module unload hdf5/1.10.7
    module unload intel-mpi/2019.8.254
    module unload libunwind
    module unload hwloc
    module unload cmake/3.18.2
    module unload boost/1.74.0
    module unload gsl/2.5
    module unload openblas/0.3.10-openmp
    module unload gcc/10.2.0
}


spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    "${SPECTRE_HOME}/support/Environments/setup/expanse_gcc.sh" "$@"
    local ret=$?
    if [ "${ret}" -ne 0 ] ; then
        echo >&2
        echo "Module setup failed!" >&2
    fi
    return "${ret}"
}

spectre_unload_modules() {
    module unload spectre_zlib
    module unload spectre_python
    module unload charm
    module unload scotch
    module unload yaml-cpp
    module unload jemalloc
    module unload libxsmm
    module unload libsharp
    module unload catch
    module unload brigand
    module unload blaze

    spectre_unload_sys_modules
}

spectre_load_modules() {
    spectre_load_sys_modules

    module load spectre_zlib
    module load spectre_python
    module load blaze
    module load brigand
    module load catch
    module load libsharp
    module load libxsmm
    module load jemalloc
    module load yaml-cpp
    module load scotch
    module load charm
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    # note that without the below gcc LFS flags, the PCH becomes
    # inconsistent with the source flags, resulting in an invalid PCH
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D DEBUG_SYMBOLS=off \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D CMAKE_CXX_FLAGS="-D_FILE_OFFSET_BITS=64 \
-D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE" \
          -D USE_SCOTCH_LB=ON \
          "$@" \
          $SPECTRE_HOME
}
