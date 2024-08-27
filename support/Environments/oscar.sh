#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    module use /oscar/data/icerm/knelli/modules
    echo "Place the following line in your '~/.bashrc' so you don't have to "
    echo "run 'spectre_setup_modules' every time you log in:"
    echo ""
    echo "module use /oscar/data/icerm/knelli/modules"
}

spectre_load_modules() {
    module purge
    module use /oscar/data/icerm/knelli/modules
    module use /users/nvu8/modules
    # Load ParaView before other Python packages, so that our Python packages
    # are higher in priority than ParaView's bundled Python packages
    module load paraview/5.11.2-osmesa
    module load spectre-deps/oscar-2024-07
}

spectre_unload_modules() {
    module use /oscar/data/icerm/knelli/modules
    module use /users/nvu8/modules
    module unload spectre-deps/oscar-2024-07
    module unload paraview/5.11.2-osmesa
    module purge
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules

    cmake -D CMAKE_C_COMPILER=clang \
          -D CMAKE_CXX_COMPILER=clang++ \
          -D CMAKE_Fortran_COMPILER=/oscar/rt/9.2/software/0.20-generic/\
0.20.1/opt/spack/linux-rhel9-x86_64_v3/gcc-11.3.1/\
gcc-13.1.0-nvrtbp3ngdnok3fg22pzxxczitvtu7ge/bin/gfortran \
          -D CCACHE_LAUNCHER_EXTRA_ENV_VARS=\
"CCACHE_CONFIGPATH=/oscar/data/icerm/knelli/spectre_cache_read.conf;"\
"CCACHE_BASEDIR=${SPECTRE_HOME}" \
          -D USE_PCH=OFF \
          -D USE_LD=gold \
          -D CHARM_ROOT=$CHARM_ROOT \
          -D BLA_VENDOR=OpenBLAS \
          -D CMAKE_BUILD_TYPE=Release \
          -D BUILD_DOCS=OFF \
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D BUILD_PYTHON_BINDINGS=ON \
          -D MACHINE=Oscar \
          -D OVERRIDE_ARCH=cascadelake \
          -D ENABLE_PARAVIEW=ON \
          "$@" \
          $SPECTRE_HOME
}
