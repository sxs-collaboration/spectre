#!/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

set -e

if [ $# != 1 ]; then
    echo "You must pass one argument to spectre_setup_modules, which"
    echo "is the directory where you want the dependencies to be built."
    exit 1
fi

# If we wanted to use AOCC (the AMD Optimizing Compiler Collection)
# we can hopefully still build the 3rd party libs with GCC. AOCC uses
# clang as its frontend and so ABI compatibility with GCC should be
# fine.
. "${SPECTRE_HOME}/support/Environments/expanse_gcc.sh"

start_dir=`pwd`
dep_dir=`realpath $1`
if [ $# != 1 ]; then
    echo "You must pass one argument to spectre_setup_modules, which"
    echo "is the directory where you want the dependencies to be built."
    return 1
fi
mkdir -p $dep_dir
cd $dep_dir

# log all output from this script
exec > >(tee "log.$(date +%F-%T)") 2>&1

mkdir -p $dep_dir/modules

spectre_load_sys_modules

if [ -f catch/include/catch.hpp ]; then
    echo "Catch is already installed"
else
    echo "Installing catch..."
    mkdir -p $dep_dir/catch/include
    cd $dep_dir/catch/include
    wget \
        https://github.com/catchorg/Catch2/releases/download/v2.13.0/catch.hpp \
        -O catch.hpp
    echo "Installed Catch into $dep_dir/catch"
    cat >$dep_dir/modules/catch <<EOF
#%Module1.0
prepend-path CPATH "$dep_dir/catch/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/catch/"
EOF
fi
cd $dep_dir

if [ -f blaze/include/blaze/Blaze.h ]; then
    echo "Blaze is already installed"
else
    echo "Installing Blaze..."
    mkdir -p $dep_dir/blaze/
    cd $dep_dir/blaze/
    wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz \
         -O blaze.tar.gz
    echo "Unpacking Blaze. This can take a few minutes..."
    tar -xzf blaze.tar.gz
    cd blaze-3.8
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D BLAZE_SHARED_MEMORY_PARALLELIZATION=NO \
          -D CMAKE_INSTALL_PREFIX=$dep_dir/blaze/ ..
    make install
    echo "Installed Blaze into $dep_dir/blaze"
    cat >$dep_dir/modules/blaze <<EOF
#%Module1.0
prepend-path CPATH "$dep_dir/blaze/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/blaze/"
EOF
fi
cd $dep_dir


if [ -f brigand/include/brigand/brigand.hpp ]; then
    echo "Brigand is already installed"
else
    echo "Installing Brigand..."
    rm -rf $dep_dir/brigand
    git clone https://github.com/edouarda/brigand.git
    echo "Installed Brigand into $dep_dir/brigand"
    cat >$dep_dir/modules/brigand <<EOF
#%Module1.0
prepend-path CPATH "$dep_dir/brigand/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/brigand/"
EOF
fi
cd $dep_dir

module use $dep_dir/modules
if [ -f libxsmm/lib/libxsmm.a ]; then
    echo "LIBXSMM is already installed"
else
    echo "Installing LIBXSMM..."
    rm -rf $dep_dir/libxsmm
    wget https://github.com/hfp/libxsmm/archive/1.16.1.tar.gz \
         -O libxsmm.tar.gz
    tar -xzf libxsmm.tar.gz
    mv libxsmm-* libxsmm
    cd libxsmm
    # Attempted an updated binutils, but does not improve matters
    # Manual settings based on a rough understanding of the available
    # vector intrinsics on the EPYC 7002 processors on Expanse
    make CXX=g++ CC=gcc FC=gfortran INTRINSICS=1 AVX=2 -j4
    cd $dep_dir
    rm libxsmm.tar.gz
    echo "Installed LIBXSMM into $dep_dir/libxsmm"
    cat >$dep_dir/modules/libxsmm <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/libxsmm/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/libxsmm/lib"
prepend-path CPATH "$dep_dir/libxsmm/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/libxsmm/"
EOF
fi
cd $dep_dir

if [ -f $dep_dir/yaml-cpp/lib/libyaml-cpp.a ]; then
    echo "yaml-cpp is already installed"
else
    echo "Installing yaml-cpp..."
    wget https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz \
         -O yaml-cpp.tar.gz
    tar -xzf yaml-cpp.tar.gz
    mv yaml-cpp-* yaml-cpp-build
    cd $dep_dir/yaml-cpp-build
    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=Release -D YAML_CPP_BUILD_TESTS=OFF \
          -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ \
          -D YAML_CPP_BUILD_CONTRIB=OFF \
          -D YAML_CPP_BUILD_TOOLS=ON \
          -D CMAKE_INSTALL_PREFIX=$dep_dir/yaml-cpp ..
    make -j4
    make install
    cd $dep_dir
    rm -r yaml-cpp-build
    rm -r yaml-cpp.tar.gz
    echo "Installed yaml-cpp into $dep_dir/yaml-cpp"
    cat >$dep_dir/modules/yaml-cpp <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/yaml-cpp/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/yaml-cpp/lib"
prepend-path CPATH "$dep_dir/yaml-cpp/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/yaml-cpp/"
EOF
fi
cd $dep_dir

if [ -f $dep_dir/libsharp/lib/libsharp.a ]; then
    echo "libsharp is already installed"
else
    echo "Installing libsharp..."
    wget https://github.com/Libsharp/libsharp/archive/v1.0.0.tar.gz \
         -O libsharp.tar.gz
    tar -xzf libsharp.tar.gz
    mv libsharp-* libsharp_build
    cd $dep_dir/libsharp_build
    autoconf
    ./configure --prefix=$dep_dir/libsharp --disable-openmp
    make -j4
    mv ./auto $dep_dir/libsharp
    cd $dep_dir
    rm -r libsharp_build
    rm libsharp.tar.gz
    echo "Installed libsharp into $dep_dir/libsharp"
    cat >$dep_dir/modules/libsharp <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/libsharp/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/libsharp/lib"
prepend-path CPATH "$dep_dir/libsharp/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/libsharp/"
EOF
fi
cd $dep_dir

if [ -f $dep_dir/scotch/lib/libscotch.a ]; then
    echo "Scotch is already installed"
else
    echo "Installing Scotch..."
    wget https://gitlab.inria.fr/scotch/scotch/-/archive/v6.1.0/scotch-v6.1.0.tar.bz2
    tar xjf scotch-v6.1.0.tar.bz2
    mv scotch-v6.1.0 scotch
    cd $dep_dir/scotch
    cp src/Make.inc/Makefile.inc.x86-64_pc_linux2 src/Makefile.inc
    cd src
    make -j4
    cd $dep_dir
    rm scotch-v6.1.0.tar.bz2
    echo "Installed Scotch into $dep_dir/scotch"
    cat >$dep_dir/modules/scotch <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/scotch/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/scotch/lib"
prepend-path CPATH "$dep_dir/scotch/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/scotch/"
EOF
fi

module use $dep_dir/modules
module load scotch
charm_version=7.0.0
charm_config=mpi-linux-x86_64-smp
# Set up Charm++ because that can be difficult
if [ -f $dep_dir/charm/$charm_config/lib/libck.a ]; then
    echo "Charm++ is already installed"
else
    echo "Installing Charm++..."
    wget https://github.com/UIUC-PPL/charm/archive/v$charm_version.tar.gz
    echo "Unpacking Charm++, this can take a few minutes"
    tar xzf v$charm_version.tar.gz
    mv charm-$charm_version charm
    cd $dep_dir/charm
    ./build LIBS $charm_config --with-production -j4
    cd $charm_config
    make ScotchLB
    cd $dep_dir
    rm v$charm_version.tar.gz
    echo "Installed Charm++ into $dep_dir/charm"
    cat >$dep_dir/modules/charm <<EOF
#%Module1.0
prepend-path PATH "$dep_dir/charm/$charm_config/bin"
prepend-path LIBRARY_PATH "$dep_dir/charm/$charm_config/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/charm/$charm_config/lib"
prepend-path CPLUS_INCLUDE_PATH "$dep_dir/charm/$charm_config/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/charm/$charm_config/"
setenv CHARM_VERSION $charm_version
setenv CHARM_HOME $dep_dir/charm/$charm_config
setenv CHARM_ROOT $dep_dir/charm/$charm_config
EOF
fi
cd $dep_dir

if [ -f $dep_dir/jemalloc/lib/libjemalloc.a ]; then
    echo "jemalloc is already set up"
else
    echo "Installing jemalloc..."
    wget https://github.com/jemalloc/jemalloc/releases/download/5.2.1/jemalloc-5.2.1.tar.bz2
    tar -xjf jemalloc-5.2.1.tar.bz2
    mv jemalloc-5.2.1 jemalloc
    cd $dep_dir/jemalloc
    ./autogen.sh
    make
    cd $dep_dir
    rm jemalloc-5.2.1.tar.bz2
    echo "Installed jemalloc into $dep_dir/jemalloc"
    cat >$dep_dir/modules/jemalloc <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/jemalloc/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/jemalloc/lib"
prepend-path CPATH "$dep_dir/jemalloc/include"
prepend-path PATH "$dep_dir/jemalloc/bin"
prepend-path LD_RUN_PATH "$dep_dir/jemalloc/lib"
setenv JEMALLOC_HOME $dep_dir/jemalloc/
prepend-path CMAKE_PREFIX_PATH "$dep_dir/jemalloc/"
EOF
fi

python3 -m venv --system-site-packages $dep_dir/py_env
export VIRTUAL_ENV=$dep_dir/py_env
export PATH=${VIRTUAL_ENV}/bin:${PATH}
cat >$dep_dir/modules/spectre_python <<EOF
#%Module1.0
setenv VIRTUAL_ENV $dep_dir/py_env
prepend-path PATH ${VIRTUAL_ENV}/bin
EOF
pip install pybind11~=2.6.1
HDF5_DIR=$HDF5HOME pip install --no-binary=h5py \
  -r $SPECTRE_HOME/support/Python/requirements.txt

cd $dep_dir

if [ -f $dep_dir/modules/spectre_zlib ]; then
    echo "zlib is already installed"
else
    echo "Installing zlib module..."
    # The zlib configuration is a bit questionable. Basically, there are
    # several different copies installed, and by default the incorrect one seems
    # to be found.
    spack_base=/cm/shared/apps/spack
    spack_root_path=$spack_base/cpu/opt/spack/linux-centos8-zen2/gcc-10.2.0/
    zlib_path=$spack_root_path/zlib-1.2.11-rchx6la4w4coybgwftagexqeqwmsqlgo/

    cat >$dep_dir/modules/spectre_zlib <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$zlib_path/lib"
prepend-path LD_LIBRARY_PATH "$zlib_path/lib"
prepend-path CPATH "$zlib_path/include"
prepend-path LD_RUN_PATH "$zlib_path/lib"
setenv ZLIB_ROOT $zlib_path/
prepend-path CMAKE_PREFIX_PATH "$zlib_path/"
EOF
fi

cd $start_dir

spectre_unload_sys_modules
module unload scotch

printf "\n\nIMPORTANT!!!\nIn order to be able to use these modules you\n"
echo "must run:"
echo "  module use $dep_dir/modules"
echo "You will need to do this every time you compile SpECTRE, so if you"
echo "want this SpECTRE environment to be your primary environment, you may"
echo "want to add it to your ~/.bashrc."
