#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Load system modules
spectre_load_sys_modules() {
    # Make sure the core default modules are loaded.
    # Without these you can get weird compilation errors
    # that make no sense.
    module load modtree/cpu libfabric zlib numactl

    module load gcc/11.2.0
    module load boost/1.74.0
    module load cmake/3.20.0
    module load gsl/2.4
    module load hdf5/1.10.7
    module load openblas/0.3.17
    module load python/3.9.5
    module unload openmpi
}

# Unload system modules
spectre_unload_sys_modules() {
    module unload boost/1.74.0
    module unload cmake/3.20.0
    module unload gsl/2.4
    module unload hdf5/1.10.7
    module unload openblas/0.3.17
    module unload python/3.9.5
    module unload gcc/11.2.0
}


spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    "${SPECTRE_HOME}/support/Environments/setup/anvil_gcc.sh" "$@"
    local ret=$?
    if [ "${ret}" -ne 0 ] ; then
        echo >&2
        echo "Module setup failed!" >&2
    fi
    return "${ret}"
}

spectre_unload_modules() {
    module unload spectre_python
    module unload charm
    module unload yaml-cpp
    module unload libxsmm
    module unload libsharp
    module unload catch
    module unload brigand
    module unload blaze
    module unload impi

    spectre_unload_sys_modules
}

spectre_load_modules() {
    spectre_load_sys_modules

    module load impi
    module load blaze
    module load brigand
    module load catch
    module load libsharp
    module load libxsmm
    module load yaml-cpp
    module load charm
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
