#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_load_modules() {
    module use /global/homes/n/nilsvu/modules
    module load cray-pmi
    module load cray-hdf5/1.12.2.9
    module load spectre-2024-04
}

spectre_unload_modules() {
    module unload cray-pmi
    module unload cray-hdf5/1.12.2.9
    module unload spectre-2024-04
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake -D CMAKE_C_COMPILER=cc \
          -D CMAKE_CXX_COMPILER=CC \
          -D CMAKE_Fortran_COMPILER=ftn \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D DEBUG_SYMBOLS=OFF \
          -D MACHINE=Perlmutter \
          "$@" \
          $SPECTRE_HOME
}
