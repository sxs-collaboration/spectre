#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Swap to GCC to 6.3.0 env
load_gcc() {
    module swap PrgEnv-cray PrgEnv-gnu
    module unload gcc
    module load gcc/6.3.0
    module load craype-hugepages8M
    module load rca
}

# Swap to Cray compiler env
load_cray_cc() {
    module unload gcc/6.3.0
    module swap PrgEnv-gnu PrgEnv-cray
}

# Since curl on BlueWaters breaks randomly we can't use spack reliably.
spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    local start_dir=`pwd`
    dep_dir=$1
    if [ $# != 1 ]; then
        echo "You must pass one argument to spectre_setup_modules, which"
        echo "is the directory where you want the dependencies to be built."
        return 1
    fi
    mkdir -p $dep_dir
    cd $dep_dir
    mkdir -p $dep_dir/modules

    load_gcc

    if [ -f catch/include/catch.hpp ]; then
        echo "Catch is already installed"
    else
        echo "Installing catch..."
        mkdir -p $dep_dir/catch/include
        cd $dep_dir/catch/include
        wget https://github.com/catchorg/Catch2/releases/download/v2.2.1/catch.hpp -O catch.hpp
        echo "Installed Catch into $dep_dir/catch"
        echo "#%Module1.0" > $dep_dir/modules/catch
        echo "prepend-path CPATH \"$dep_dir/catch/include\"" >> $dep_dir/modules/catch
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/catch/\"" >> $dep_dir/modules/catch
    fi
    cd $dep_dir

    if [ -f blaze/include/blaze/Blaze.h ]; then
        echo "Blaze is already installed"
    else
        echo "Installing Blaze..."
        mkdir -p $dep_dir/blaze/
        cd $dep_dir/blaze/
        wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.2.tar.gz -O blaze.tar.gz
        tar -xzf blaze.tar.gz
        mv blaze-* include
        echo "Installed Blaze into $dep_dir/blaze"
        echo "#%Module1.0" > $dep_dir/modules/blaze
        echo "prepend-path CPATH \"$dep_dir/blaze/include\"" >> $dep_dir/modules/blaze
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/blaze/\"" >> $dep_dir/modules/blaze
    fi
    cd $dep_dir


    if [ -f brigand/include/brigand/brigand.hpp ]; then
        echo "Brigand is already installed"
    else
        echo "Installing Brigand..."
        rm -rf $dep_dir/brigand
        git clone https://github.com/edouarda/brigand.git
        echo "Installed Brigand into $dep_dir/brigand"
        echo "#%Module1.0" > $dep_dir/modules/brigand
        echo "prepend-path CPATH \"$dep_dir/brigand/include\"" >> $dep_dir/modules/brigand
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/brigand/\"" >> $dep_dir/modules/brigand
    fi
    cd $dep_dir

    if [ -f libxsmm/lib/libxsmm.a ]; then
        echo "LIBXSMM is already installed"
    else
        load_gcc
        echo "Installing LIBXSMM..."
        rm -rf $dep_dir/libxsmm
        wget https://github.com/hfp/libxsmm/archive/1.9.tar.gz -O libxsmm.tar.gz
        tar -xzf libxsmm.tar.gz
        mv libxsmm-* libxsmm
        cd libxsmm
        make CXX=CC CC=cc FC=ftn -j4
        cd $dep_dir
        rm libxsmm.tar.gz
        echo "Installed LIBXSMM into $dep_dir/libxsmm"
        echo "#%Module1.0" > $dep_dir/modules/libxsmm
        echo "prepend-path LIBRARY_PATH \"$dep_dir/libxsmm/lib\"" >> $dep_dir/modules/libxsmm
        echo "prepend-path LD_LIBRARY_PATH \"$dep_dir/libxsmm/lib\"" >> $dep_dir/modules/libxsmm
        echo "prepend-path CPATH \"$dep_dir/libxsmm/include\"" >> $dep_dir/modules/libxsmm
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/libxsmm/\"" >> $dep_dir/modules/libxsmm
    fi
    cd $dep_dir

    if [ -f jemalloc/include/jemalloc/jemalloc.h ]; then
        echo "jemalloc is already installed"
    else
        echo "Installing jemalloc..."
        rm -r $dep_dir/jemalloc
        wget https://github.com/jemalloc/jemalloc/archive/5.0.1.tar.gz -O jemalloc.tar.gz
        tar -xzf jemalloc.tar.gz
        mv jemalloc-* jemalloc-build
        cd $dep_dir/jemalloc-build
        module load autoconf/2.69
        ./autogen.sh --prefix=$dep_dir/jemalloc
        make -j4
        make install_bin && make install_include && make install_lib
        cd $dep_dir
        rm -r $dep_dir/jemalloc-build
        rm $dep_dir/jemalloc.tar.gz
        module unload autoconf/2.69
        echo "Installed jemalloc into $dep_dir/jemalloc"
        echo "#%Module1.0" > $dep_dir/modules/jemalloc
        echo "prepend-path LIBRARY_PATH \"$dep_dir/jemalloc/lib\"" >> $dep_dir/modules/jemalloc
        echo "prepend-path LD_LIBRARY_PATH \"$dep_dir/jemalloc/lib\"" >> $dep_dir/modules/jemalloc
        echo "prepend-path CPATH \"$dep_dir/jemalloc/include\"" >> $dep_dir/modules/jemalloc
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/jemalloc/\"" >> $dep_dir/modules/jemalloc
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
        module load cmake/3.9.4
        cmake -D CMAKE_BUILD_TYPE=Release -D YAML_CPP_BUILD_TESTS=OFF \
              -D YAML_CPP_BUILD_CONTRIB=OFF \
              -D YAML_CPP_BUILD_TOOLS=ON \
              -D CMAKE_INSTALL_PREFIX=$dep_dir/yaml-cpp ..
        make -j4
        make install
        module unload cmake/3.9.4
        cd $dep_dir
        rm -r yaml-cpp-build
        rm -r yaml-cpp.tar.gz
        echo "Installed yaml-cpp into $dep_dir/yaml-cpp"
        echo "#%Module1.0" > $dep_dir/modules/yaml-cpp
        echo "prepend-path LIBRARY_PATH \"$dep_dir/yaml-cpp/lib\"" >> $dep_dir/modules/yaml-cpp
        echo "prepend-path LD_LIBRARY_PATH \"$dep_dir/yaml-cpp/lib\"" >> $dep_dir/modules/yaml-cpp
        echo "prepend-path CPATH \"$dep_dir/yaml-cpp/include\"" >> $dep_dir/modules/yaml-cpp
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/yaml-cpp/\"" >> $dep_dir/modules/yaml-cpp
    fi
    cd $dep_dir

    # Set up Charm++ because that can be difficult
    if [ -f $dep_dir/charm/gni-crayxe-smp/lib/libck.a ]; then
        echo "Charm++ is already installed"
    else
        echo "Installing Charm++..."
        load_gcc
        git clone https://charm.cs.illinois.edu/gerrit/charm
        cd $dep_dir/charm
        git checkout v6.8.2
        ./build charm++ gni-crayxe smp -j4 --with-production
        git apply $SPECTRE_HOME/support/Charm/v6.8.patch
        cd $dep_dir
        rm charm.tar.gz
        echo "Installed Charm++ into $dep_dir/charm"
        echo "#%Module1.0" > $dep_dir/modules/charm
        echo "prepend-path LIBRARY_PATH \"$dep_dir/charm/lib\"" >> $dep_dir/modules/charm
        echo "prepend-path LD_LIBRARY_PATH \"$dep_dir/charm/lib\"" >> $dep_dir/modules/charm
        echo "prepend-path CPATH \"$dep_dir/charm/include\"" >> $dep_dir/modules/charm
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/charm/\"" >> $dep_dir/modules/charm
        echo "setenv CHARM_VERSION 6.8.2" >> $dep_dir/modules/charm
        echo "setenv CHARM_HOME $dep_dir/charm/gni-crayxe-smp" >> $dep_dir/modules/charm
        echo "setenv CHARM_ROOT $dep_dir/charm/gni-crayxe-smp" >> $dep_dir/modules/charm
    fi
    cd $dep_dir

    # We need to be able to link in the MKL, which BlueWaters doesn't allow
    echo "#%Module1.0" > $dep_dir/modules/mkl
    echo "prepend-path LIBRARY_PATH \"/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64_lin\"" >> $dep_dir/modules/mkl
    echo "prepend-path LD_LIBRARY_PATH \"/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64_lin\"" >> $dep_dir/modules/mkl
    echo "prepend-path CPATH \"/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/include\"" >> $dep_dir/modules/mkl
    echo "prepend-path CMAKE_PREFIX_PATH \"/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/\"" >> $dep_dir/modules/mkl

    # Set up env variables needed to get traces and linking working
    echo "#%Module1.0" > $dep_dir/modules/env_vars
    echo "setenv XTPE_LINK_TYPE dynamic" >> $dep_dir/modules/env_vars
    echo "setenv CRAYPE_LINK_TYPE dynamic" >> $dep_dir/modules/env_vars
    echo "setenv CRAY_ADD_RPATH yes" >> $dep_dir/modules/env_vars
    echo "setenv XTPE_LINK_TYPE dynamic" >> $dep_dir/modules/env_vars
    echo "prepend-path PE_PKGCONFIG_LIBS \"cray-pmi\"" >> $dep_dir/modules/env_vars
    echo "prepend-path PE_PKGCONFIG_LIBS \"cray-ugni\"" >> $dep_dir/modules/env_vars
    echo "setenv ATP_ENABLED 1" >> $dep_dir/modules/env_vars

    cd $start_dir

    printf "\n\nIMPORTANT!!!\nIn order to be able to use these modules you\n"
    echo "must run:"
    echo "  module use $dep_dir/modules"
    echo "You will need to do this every time you compile SpECTRE, so you may"
    echo "want to add it to your ~/.bashrc."
}

spectre_unload_modules() {
    load_cray_cc
    # Unload system modules
    module unload rca
    module unload craype-hugepages8M
    module unload gsl
    module unload cray-hdf5/1.8.16
    module unload bwpy
    export -n SPECTRE_BOOST_ROOT
    export -n BOOST_ROOT

    # Unload user modules
    module unload blaze
    module unload brigand
    module unload catch
    module unload charm
    module unload jemalloc
    module unload libxsmm
    module unload yaml-cpp

    module unload mkl
    module unload env_vars
    ulimit -c unlimited
}

spectre_load_modules() {
    echo "If you receive module not found errors make sure to run"
    echo "  module use /PATH/TO/DEPS/modules"
    echo "and then load the modules"
    echo "Unloading any existing SpECTRE-related modules"
    spectre_unload_modules
    echo "Loading SpECTRE-related modules"
    load_gcc
    # Load system modules, the order matters
    # The boost installation is not quite right, so we need to work around it
    module load boost/1.63.0
    export SPECTRE_BOOST_ROOT=$BOOST_ROOT
    module unload boost/1.63.0
    export BOOST_ROOT=$SPECTRE_BOOST_ROOT
    module load craype-hugepages8M
    module load gsl
    module load cray-hdf5/1.8.16
    module load bwpy

    # Load user modules
    module load blaze
    module load brigand
    module load catch
    module load charm
    module load jemalloc
    module load libxsmm
    module load yaml-cpp

    module load mkl
    module load env_vars
    ulimit -c unlimited
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_CXX_COMPILER=CC \
          -D CMAKE_C_COMPILER=cc \
          -D CMAKE_Fortran_COMPILER=ftn \
          -D Boost_USE_STATIC_LIBS=ON \
          -D Boost_USE_STATIC_RUNTIME=ON \
          -D USE_SYSTEM_INCLUDE=OFF \
          -D ENABLE_WARNINGS=OFF \
          -D CMAKE_EXE_LINKER_FLAGS="-I$BOOST_ROOT/include" \
          -D USE_PCH=OFF \
          $SPECTRE_HOME
}
