#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    echo "All modules on zwicky are provided by the system"
}

spectre_unload_modules() {
module unload charm
module unload catch-2.2.1-gcc-7.3.0-yaw3hxt
module unload perl/5.26.2
module unload python-2.7.15-gcc-7.3.0-qvsb5jh
module unload py-numpy-1.14.3-gcc-7.3.0-fgkjppi
module unload spack
module unload gcc/7.3.0
module unload openmpi/3.1.0
module unload cmake/3.11.4
module unload binutils-2.29.1-gcc-7.3.0-v6il5pu
module unload blaze-3.2-gcc-7.3.0-m6qrjsx
module unload boost-1.67.0-gcc-7.3.0-bpgr5si
module unload brigand-master-gcc-7.3.0-7mqn34k
module unload gsl-2.4-gcc-7.3.0-fkcmpxl
module unload hdf5-1.10.2-gcc-7.3.0-2excv3j
module unload jemalloc-4.5.0-gcc-7.3.0-adzurmv
module unload libxsmm/master
module unload curl-7.60.0-gcc-7.3.0-dbnl246
module unload openssl-1.0.2n-gcc-7.3.0-mtsyoqu
module unload openblas-0.3.0-gcc-7.3.0-vp5upid
module unload yaml-cpp/develop
module unload git-2.17.1-gcc-7.3.0-nmqkxud
module unload openssh-7.6p1-gcc-7.3.0-gg6cb3b
}

spectre_load_modules() {
module load charm
module load catch-2.2.1-gcc-7.3.0-yaw3hxt
module load perl/5.26.2
module load python-2.7.15-gcc-7.3.0-qvsb5jh
module load py-numpy-1.14.3-gcc-7.3.0-fgkjppi
module load spack
module load gcc/7.3.0
module load openmpi/3.1.0
module load cmake/3.11.4
module load binutils-2.29.1-gcc-7.3.0-v6il5pu
module load blaze-3.2-gcc-7.3.0-m6qrjsx
module load boost-1.67.0-gcc-7.3.0-bpgr5si
module load brigand-master-gcc-7.3.0-7mqn34k
module load gsl-2.4-gcc-7.3.0-fkcmpxl
module load hdf5-1.10.2-gcc-7.3.0-2excv3j
module load jemalloc-4.5.0-gcc-7.3.0-adzurmv
module load libxsmm/master
module load curl-7.60.0-gcc-7.3.0-dbnl246
module load openssl-1.0.2n-gcc-7.3.0-mtsyoqu
module load openblas-0.3.0-gcc-7.3.0-vp5upid
module load yaml-cpp/develop
module load git-2.17.1-gcc-7.3.0-nmqkxud
module load openssh-7.6p1-gcc-7.3.0-gg6cb3b
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
          $SPECTRE_HOME
}
