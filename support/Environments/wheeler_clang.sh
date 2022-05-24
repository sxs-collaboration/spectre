#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# If Intel MPI gets updated or Charm++ changes the way it builds MPI
# configurations we might be able to enable clang again.
echo "Cannot use Clang with Intel MPI v2017.1."
return 1

spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    local start_dir=`pwd`
    dep_dir=`realpath $1`
    if [ $# != 1 ]; then
        echo "You must pass one argument to spectre_setup_modules, which"
        echo "is the directory where you want the dependencies to be built."
        return 1
    fi
    mkdir -p $dep_dir
    cd $dep_dir
    mkdir -p $dep_dir/modules

    cd $dep_dir

    if [ -f $dep_dir/arpack/lib64/libarpack.a ]; then
        echo "arpack is already set up"
    else
        echo "Installing arpack..."
        wget https://github.com/opencollab/arpack-ng/archive/refs/tags/3.8.0.tar.gz
        tar -xzf 3.8.0.tar.gz
        mv arpack-ng-3.8.0 arpack-build
        cd $dep_dir/arpack-build
        mkdir build
        cd build
        cmake -D CMAKE_BUILD_TYPE=Release \
              -D CMAKE_C_COMPILER=gcc \
              -D BUILD_SHARED_LIBS=OFF \
              -D CMAKE_INSTALL_PREFIX=$dep_dir/arpack ..
        make -j4
        make install
        cd $dep_dir
        rm 3.8.0.tar.gz
        rm -r arpack-build
        echo "Installed arpack into $dep_dir/arpack"
        cat >$dep_dir/modules/arpack <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/arpack/lib64"
prepend-path LD_LIBRARY_PATH "$dep_dir/arpack/lib64"
prepend-path CPATH "$dep_dir/arpack/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/arpack/"
EOF
    fi

    cd $start_dir

    printf "\n\nIMPORTANT!!!\nIn order to be able to use these modules you\n"
    echo "must run:"
    echo "  module use $dep_dir/modules"
    echo "You will need to do this every time you compile SpECTRE, so you may"
    echo "want to add it to your ~/.bashrc."

}

spectre_unload_modules() {
    module unload gcc/7.3.0
    module unload blaze/3.8
    module unload boost/1.65.0-gcc-6.4.0
    module unload brigand/master
    module unload catch/2.13.3
    module unload gsl/2.1
    module unload libsharp/1.0.0
    module unload libxsmm/1.16.1
    module unload openblas/0.2.18
    module unload papi/5.5.1
    module unload yaml-cpp/master
    module unload impi/2017.1
    module unload cmake/3.18.2
    module unload ninja/1.10.0
    module unload doxygen/1.8.13
    module unload git/2.8.4
    module unload llvm/10.0.0
    module unload charm/7.0.0-intelmpi-smp
    module unload python/anaconda3-2019.10
    module unload pybind11/2.6.1
    module unload arpack
}

spectre_load_modules() {
    module load gcc/7.3.0
    module load blaze/3.8
    module load boost/1.65.0-gcc-6.4.0
    module load brigand/master
    module load catch/2.13.3
    module load gsl/2.1
    module load libsharp/1.0.0
    module load libxsmm/1.16.1
    module load openblas/0.2.18
    module load papi/5.5.1
    module load yaml-cpp/master
    module load impi/2017.1
    module load cmake/3.18.2
    module load ninja/1.10.0
    module load doxygen/1.8.13
    module load git/2.8.4
    module load llvm/10.0.0
    module load charm/7.0.0-intelmpi-smp
    module load python/anaconda3-2019.10
    module load pybind11/2.6.1
    module load arpack
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    # Notes:
    # - Set CMAKE_PREFIX_PATH to pick up packages consistent with the anaconda
    #   module, such as zlib. The anaconda module on Wheeler does not set this
    #   automatically.
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_C_COMPILER=clang \
          -D CMAKE_CXX_COMPILER=clang++ \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D MEMORY_ALLOCATOR=SYSTEM \
          -D BUILD_PYTHON_BINDINGS=OFF \
          -D CMAKE_PREFIX_PATH="$PYTHON_HOME" \
          "$@" \
          $SPECTRE_HOME
}
