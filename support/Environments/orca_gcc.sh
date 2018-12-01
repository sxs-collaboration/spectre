#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    echo "All modules on orca are provided by the system"
}

spectre_unload_modules() {
module unload python/2.7.15
module unload spack
module unload gcc/7.3.0
module unload openmpi/3.1.0
module unload blaze-3.2-gcc-7.3.0-u6l3rrb
module unload boost-1.67.0-gcc-7.3.0-e6bshzw
module unload brigand-master-gcc-7.3.0-zizme5m
module unload catch-2.2.1-gcc-7.3.0-fhnfleh
module unload gsl-2.4-gcc-7.3.0-okfp45i
module unload hdf5-1.10.2-gcc-7.3.0-7erzlsy
module unload jemalloc-4.5.0-gcc-7.3.0-crqk7yv
module unload openblas-0.2.20-gcc-7.3.0-rjpf3cb
module unload yaml-cpp-develop-gcc-7.3.0-wy6dpng
module unload libxsmm/1.9
module unload cmake-3.11.3-gcc-7.3.0-isxbgez
module unload charm
module unload zlib-1.2.11-gcc-7.3.0-qzvtd4j
module unload libsharp/1.0.0
}

spectre_load_modules() {
module load python/2.7.15
module load spack
module load gcc/7.3.0
module load openmpi/3.1.0
module load blaze-3.2-gcc-7.3.0-u6l3rrb
module load boost-1.67.0-gcc-7.3.0-e6bshzw
module load brigand-master-gcc-7.3.0-zizme5m
module load catch-2.2.1-gcc-7.3.0-fhnfleh
module load gsl-2.4-gcc-7.3.0-okfp45i
module load hdf5-1.10.2-gcc-7.3.0-7erzlsy
module load jemalloc-4.5.0-gcc-7.3.0-crqk7yv
module load openblas-0.2.20-gcc-7.3.0-rjpf3cb
module load yaml-cpp-develop-gcc-7.3.0-wy6dpng
module load libxsmm/1.9
module load cmake-3.11.3-gcc-7.3.0-isxbgez
module load charm
module load zlib-1.2.11-gcc-7.3.0-qzvtd4j
module load libsharp/1.0.0
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
          -D PYTHON_EXECUTABLE=/share/apps/python/2.7.15/bin/python \
          $SPECTRE_HOME
}
