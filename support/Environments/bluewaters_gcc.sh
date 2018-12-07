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
        wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.2.tar.gz -O blaze.tar.gz
        tar -xzf blaze.tar.gz
        mv blaze-* include
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
        cat >$dep_dir/modules/libxsmm <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/libxsmm/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/libxsmm/lib"
prepend-path CPATH "$dep_dir/libxsmm/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/libxsmm/"
EOF
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
        cat >$dep_dir/modules/jemalloc <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/jemalloc/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/jemalloc/lib"
prepend-path CPATH "$dep_dir/jemalloc/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/jemalloc/"
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
prepend-path CPATH "$dep_dir/libsharp/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/libsharp/"
EOF
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
        cat >$dep_dir/modules/charm <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/charm/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/charm/lib"
prepend-path CPATH "$dep_dir/charm/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/charm/"
setenv CHARM_VERSION 6.8.2
setenv CHARM_HOME $dep_dir/charm/gni-crayxe-smp
setenv CHARM_ROOT $dep_dir/charm/gni-crayxe-smp
EOF
    fi
    cd $dep_dir

    # We need to be able to link in the MKL, which BlueWaters doesn't allow
    MKL_DIR=/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl
    cat >$dep_dir/modules/mkl <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "${MKL_DIR}/lib/intel64_lin"
prepend-path LD_LIBRARY_PATH "${MKL_DIR}/lib/intel64_lin"
prepend-path CPATH "${MKL_DIR}/include"
prepend-path CMAKE_PREFIX_PATH "${MKL_DIR}/"
EOF

    # Set up env variables needed to get traces and linking working
    cat >$dep_dir/modules/env_vars <<EOF
#%Module1.0
setenv XTPE_LINK_TYPE dynamic
setenv CRAYPE_LINK_TYPE dynamic
setenv CRAY_ADD_RPATH yes
setenv XTPE_LINK_TYPE dynamic
prepend-path PE_PKGCONFIG_LIBS "cray-pmi"
prepend-path PE_PKGCONFIG_LIBS "cray-ugni"
setenv ATP_ENABLED 1
EOF

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
    module unload libsharp
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
    module load libsharp
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
