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
    source /home/geoffrey/apps/spack/share/spack/setup-env.sh
    spack unload cmake@3.14.1
    spack unload git@2.21.0
    spack unload blaze@3.2
    spack unload brigand@master
    spack unload libsharp@2018-01-17
    spack unload catch@2.6.1
    spack unload gsl@2.5
    spack unload jemalloc@4.5.0
    spack unload libxsmm@1.10
    spack unload yaml-cpp@develop
    spack unload boost@1.69.0
    spack unload hdf5@1.10.5~hl
    spack unload openblas@0.3.5
    module unload orca0/charm
    module unload orca0/python/3.7.0
}

spectre_load_modules() {
    module purge
    export MODULEPATH=/home/geoffrey/apps/modules:$MODULEPATH
    module load ohpc
    source /home/geoffrey/apps/spack/share/spack/setup-env.sh
    spack load cmake@3.14.1
    spack load git@2.21.0
    spack load blaze@3.2
    spack load brigand@master
    spack load libsharp@2018-01-17
    spack load catch@2.6.1
    spack load gsl@2.5
    spack load jemalloc@4.5.0
    spack load libxsmm@1.10
    spack load yaml-cpp@develop
    spack load boost@1.69.0
    spack load hdf5@1.10.5~hl
    spack load openblas@0.3.5
    module load orca0/charm
    module load orca0/python/3.7.0
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          $SPECTRE_HOME
}
