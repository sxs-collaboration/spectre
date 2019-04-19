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
    module unload gnu7/7.3.0
    module unload openmpi/1.10.7
    module unload prun/1.2
    module unload cmake-3.13.1-gcc-7.3.0-r7qr3qo
    module unload git-2.19.2-gcc-7.3.0-jfnpgdh
    module unload blaze-3.2-gcc-7.3.0-d4xgiej
    module unload brigand-master-gcc-7.3.0-3m5ibui
    module unload libsharp-2018-01-17-gcc-7.3.0-4xamgaw
    module unload catch-2.4.0-gcc-7.3.0-prvl6kv
    module unload gsl-2.5-gcc-7.3.0-i7icadp
    module unload jemalloc-4.5.0-gcc-7.3.0-wlf2m7r
    module unload libxsmm-1.10-gcc-7.3.0-sjh5yzv
    module unload yaml-cpp-develop-gcc-7.3.0-qcfbbll
    module unload boost-1.68.0-gcc-7.3.0-vgl6ofr
    module unload hdf5-1.10.4-gcc-7.3.0-ytt4j54
    module unload openblas-0.3.4-gcc-7.3.0-tt2coe7
    module unload python/3.7.0
    module unload charm
}

spectre_load_modules() {
    module load ohpc
    module load python/3.7.0
    module load gnu7/7.3.0
    module load openmpi/1.10.7
    module load prun/1.2
    source /opt/ohpc/pub/apps/spack/0.12.0/share/spack/setup-env.sh
    module load cmake-3.13.1-gcc-7.3.0-r7qr3qo
    module load git-2.19.2-gcc-7.3.0-jfnpgdh
    module load blaze-3.2-gcc-7.3.0-d4xgiej
    module load brigand-master-gcc-7.3.0-3m5ibui
    module load libsharp-2018-01-17-gcc-7.3.0-4xamgaw
    module load catch-2.4.0-gcc-7.3.0-prvl6kv
    module load gsl-2.5-gcc-7.3.0-i7icadp
    module load jemalloc-4.5.0-gcc-7.3.0-wlf2m7r
    module load libxsmm-1.10-gcc-7.3.0-sjh5yzv
    module load yaml-cpp-develop-gcc-7.3.0-qcfbbll
    module load boost-1.68.0-gcc-7.3.0-vgl6ofr
    module load hdf5-1.10.4-gcc-7.3.0-ytt4j54
    module load openblas-0.3.4-gcc-7.3.0-tt2coe7
    module load charm
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
