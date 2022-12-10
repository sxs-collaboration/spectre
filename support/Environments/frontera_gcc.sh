#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Load system modules
spectre_load_sys_modules() {
    # impi is loaded, which it should be by default
    # but explicitly load it in just in case
    module load impi/19.0.9
    module load gcc/9.1.0
    module load mkl/19.0.5
    module load gsl
    module load hdf5
    module load boost
}

# Unload system modules
spectre_unload_sys_modules() {
    module unload boost
    module unload hdf5
    module unload gsl
    module unload mkl/19.0.5
    module unload gcc/9.1.0
    # Don't unload impi as this is one of the default system modules
}


spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    "${SPECTRE_HOME}/support/Environments/setup/frontera_gcc.sh" "$@"
    local ret=$?
    if [ "${ret}" -ne 0 ] ; then
        echo >&2
        echo "Module setup failed!" >&2
    fi
    return "${ret}"
}

spectre_unload_modules() {
    module unload spectre_python
    module unload charm_mpi
    module unload yaml-cpp
    module unload spectre_boost
    module unload libxsmm
    module unload libsharp
    module unload catch
    module unload brigand
    module unload blaze

    spectre_unload_sys_modules
}

spectre_load_modules() {
    spectre_load_sys_modules

    module load blaze
    module load brigand
    module load catch
    module load libsharp
    module load libxsmm
    module load spectre_boost
    module load yaml-cpp
    module load charm_mpi
    module load spectre_python
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    # -D USE_LD=ld - ld.gold seems to hang linking the main executables
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D Python_EXECUTABLE=`which python3` \
          -D USE_LD=ld \
          -D SPECTRE_TEST_RUNNER="$(pwd)/bin/charmrun" \
          "$@" \
          $SPECTRE_HOME
}
