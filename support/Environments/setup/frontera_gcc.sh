#!/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

set -e

. "${SPECTRE_HOME}/support/Environments/frontera_gcc.sh"

if [ $# != 1 ]; then
    echo "You must pass one argument to spectre_setup_modules, which"
    echo "is the directory where you want the dependencies to be built."
    exit 1
fi

dep_dir=`realpath $1`
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
    wget https://github.com/catchorg/Catch2/releases/download/v2.13.0/catch.hpp -O catch.hpp
    echo "Installed Catch into $dep_dir/catch"
cat >$dep_dir/modules/catch <<EOF
#%Module1.0
prepend-path CPLUS_INCLUDE_PATH "$dep_dir/catch/include"
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
    wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz -O blaze.tar.gz
    tar -xzf blaze.tar.gz
    mv blaze-* include
    echo "Installed Blaze into $dep_dir/blaze"
    cat >$dep_dir/modules/blaze <<EOF
#%Module1.0
prepend-path CPLUS_INCLUDE_PATH "$dep_dir/blaze/include"
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
prepend-path CPLUS_INCLUDE_PATH "$dep_dir/brigand/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/brigand/"
EOF
fi
cd $dep_dir

if [ -f libxsmm/lib/libxsmm.a ]; then
    echo "LIBXSMM is already installed"
else
    echo "Installing LIBXSMM..."
    rm -rf $dep_dir/libxsmm
    wget https://github.com/hfp/libxsmm/archive/1.16.1.tar.gz -O libxsmm.tar.gz
    tar -xzf libxsmm.tar.gz
    mv libxsmm-* libxsmm
    cd libxsmm
    make CXX=g++ CC=gcc FC=gfortran -j4
    cd $dep_dir
    rm libxsmm.tar.gz
    echo "Installed LIBXSMM into $dep_dir/libxsmm"
    cat >$dep_dir/modules/libxsmm <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/libxsmm/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/libxsmm/lib"
prepend-path C_INCLUDE_PATH "$dep_dir/libxsmm/include"
prepend-path CPLUS_INCLUDE_PATH "$dep_dir/libxsmm/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/libxsmm/"
EOF
fi
cd $dep_dir

if [ -f $dep_dir/yaml-cpp/lib/libyaml-cpp.a ]; then
    echo "yaml-cpp is already installed"
else
    echo "Installing yaml-cpp..."
    wget https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz -O yaml-cpp.tar.gz
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
prepend-path CPLUS_INCLUDE_PATH "$dep_dir/yaml-cpp/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/yaml-cpp/"
EOF
fi
cd $dep_dir

if [ -f $dep_dir/libsharp/lib/libsharp.a ]; then
    echo "libsharp is already installed"
else
    echo "Installing libsharp..."
    wget https://github.com/Libsharp/libsharp/archive/v1.0.0.tar.gz -O libsharp.tar.gz
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
prepend-path CPLUS_INCLUDE_PATH "$dep_dir/libsharp/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/libsharp/"
EOF
fi
cd $dep_dir

# Set up Charm++ because that can be difficult
charm_version=7.0.0
backend=mpi
charm_config=${backend}-linux-x86_64-smp
if [ -f $dep_dir/charm/${charm_config}/lib/libck.a ]; then
    echo "Charm++ is already installed"
else
    echo "Installing Charm++..."
    wget https://github.com/UIUC-PPL/charm/archive/v${charm_version}.tar.gz
    tar xzf v${charm_version}.tar.gz
    mv charm-${charm_version} charm
    cd $dep_dir/charm
    if [ -f ${SPECTRE_HOME}/support/Charm/v${charm_version}.patch ]; then
        git apply ${SPECTRE_HOME}/support/Charm/v${charm_version}.patch
    fi
    ./build LIBS ${charm_config} --with-production -j6
    cd $dep_dir
    rm v${charm_version}.tar.gz
    echo "Installed Charm++ into $dep_dir/charm"
    cat >$dep_dir/modules/charm_${backend} <<EOF
#%Module1.0
prepend-path PATH "$dep_dir/charm/${charm_config}/bin"
prepend-path LIBRARY_PATH "$dep_dir/charm/${charm_config}/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/charm/${charm_config}/lib"
prepend-path CPLUS_INCLUDE_PATH "$dep_dir/charm/${charm_config}/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/charm/${charm_config}/"
setenv CHARM_VERSION ${charm_version}
setenv CHARM_HOME $dep_dir/charm/${charm_config}
setenv CHARM_ROOT $dep_dir/charm/${charm_config}
EOF
fi
cd $dep_dir

if [ -f $dep_dir/modules/spectre_boost ]; then
    echo "Boost is already set up"
else
    cat >$dep_dir/modules/spectre_boost <<EOF
#%Module1.0
prepend-path CPLUS_INCLUDE_PATH "$BOOST_ROOT/include"
EOF
fi
cd $dep_dir

python3 -m venv --system-site-packages $dep_dir/py_env
export VIRTUAL_ENV=$dep_dir/py_env
export PATH=${VIRTUAL_ENV}/bin:${PATH}
cat >$dep_dir/modules/spectre_python <<EOF
#%Module1.0
setenv VIRTUAL_ENV $dep_dir/py_env
prepend-path PATH ${VIRTUAL_ENV}/bin
EOF
pip install pybind11~=2.6.1
HDF5_DIR=$TACC_HDF5_DIR pip install --no-binary=h5py \
  -r $SPECTRE_HOME/support/Python/requirements.txt

printf "\n\nIMPORTANT!!!\nIn order to be able to use these modules you\n"
echo "must run:"
echo "  module use $dep_dir/modules"
echo "You will need to do this every time you compile SpECTRE, so you may"
echo "want to add it to your ~/.bashrc."
