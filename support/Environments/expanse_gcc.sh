#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Load system modules
spectre_load_sys_modules() {
    module load gcc/10.2.0
    module load intel-mkl/2019.1.144
    module load gsl/2.5
    module load boost/1.74.0
    module load cmake/3.18.2
}

# Unload system modules
spectre_unload_sys_modules() {
    module unload cmake/3.18.2
    module unload boost/1.74.0
    module unload gsl/2.5
    module unload intel-mkl/2019.1.1
    module unload gcc/10.2.0
}


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

    spectre_load_sys_modules

    if [ -f hdf5-install/usr/local/HDF_Group/HDF5/1.12.0/lib/libhdf5.a ]; then
        echo "HDF5 is already installed"
    else
        echo "Installing hdf5..."
        rm -rf $dep_dir/hdf5
        rm -rf $dep_dir/hdf5-install
        git clone --depth 1 --branch hdf5-1_12_0 \
            https://github.com/HDFGroup/hdf5.git
        mkdir hdf5-install
        cd $dep_dir/hdf5
        mkdir build
        cd build
        cmake --config Release $dep_dir/hdf5
        make -j4
        make install DESTDIR=$dep_dir/hdf5-install
        echo "Installed HDF5 to $dep_dir/hdf5-install/"
        hdf5_prefix=$dep_dir/hdf5-install/usr/local/HDF_Group/HDF5/1.12.0
        cat >$dep_dir/modules/hdf5 <<EOF
#%Module1.0
prepend-path PATH "$hdf5_prefix/bin/"
setenv HDF5_DIR "$hdf5_prefix"
prepend-path LIBRARY_PATH "$hdf5_prefix/lib"
prepend-path LD_LIBRARY_PATH "$hdf5_prefix/lib"
prepend-path CPATH "$hdf5_prefix/include"
EOF
    fi
    cd $dep_dir

    # These would be available via the anaconda3 module, but anaconda
    # brings in openssl version that causes clashes with ssh operations
    # performed by charm++ that we have not yet found a resolution for.
    # We also have to --ignore-installed on the numpy install because
    # otherwise pip will insist that we use the (broken) install
    # already on expanse...
    pip3 install --user --ignore-installed numpy
    pip3 install --user scipy
    pip3 install --user matplotlib
    pip3 install --user h5py

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
        wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz -O blaze.tar.gz
        tar -xzf blaze.tar.gz
        mv blaze-* include
        ln -s include/blaze-3.8/blaze include/blaze
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
        wget https://github.com/hfp/libxsmm/archive/1.16.1.tar.gz -O libxsmm.tar.gz
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
    # Set up Charm++ because that can be difficult
    if [ -f $dep_dir/charm/verbs-linux-x86_64-smp/lib/libck.a ]; then
        echo "Charm++ is already installed"
    else
        echo "Installing Charm++..."
        wget https://github.com/UIUC-PPL/charm/archive/v6.10.2.tar.gz
        tar xzf v6.10.2.tar.gz
        mv charm-6.10.2 charm
        cd $dep_dir/charm
        ./build charm++ verbs-linux-x86_64-smp --with-production -j4
        ./build EveryLB verbs-linux-x86_64-smp --with-production -j4
        ./build ScotchLB verbs-linux-x86_64-smp --with-production -j4
        cd $dep_dir
        rm v6.10.2.tar.gz
        echo "Installed Charm++ into $dep_dir/charm"
        cat >$dep_dir/modules/charm <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/charm/verbs-linux-x86_64-smp/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/charm/verbs-linux-x86_64-smp/lib"
prepend-path CPATH "$dep_dir/charm/verbs-linux-x86_64-smp/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/charm/verbs-linux-x86_64-smp/"
prepend-path PATH "$dep_dir/charm/verbs-linux-x86_64-smp/bin"
setenv CHARM_VERSION 6.10.2
setenv CHARM_HOME $dep_dir/charm/verbs-linux-x86_64-smp
setenv CHARM_ROOT $dep_dir/charm/verbs-linux-x86_64-smp
EOF
    fi
    cd $dep_dir

    if [ -f $dep_dir/modules/spectre_gsl ]; then
        echo "GSL is already set up"
    else
        # This hard-coded gsl path is not easily maintainable, so if and when
        # Caltech HPC assigns an accessible environment variable for the loaded
        # GSL module, we should use that in favor of the hard-coded path
        cat >$dep_dir/modules/spectre_gsl <<EOF
#%Module1.0
setenv GSL_HOME "/software/gsl/2.4"
setenv GSL_ROOT "/software/gsl/2.4"
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

    cd $start_dir

    spectre_unload_sys_modules
    module unload scotch
    module unload binutils

    printf "\n\nIMPORTANT!!!\nIn order to be able to use these modules you\n"
    echo "must run:"
    echo "  module use $dep_dir/modules"
    echo "You will need to do this every time you compile SpECTRE, so if you"
    echo "want this SpECTRE environment to be your primary environment, you may"
    echo "want to add it to your ~/.bashrc."
}

spectre_unload_modules() {
    module unload charm
    module unload scotch
    module unload yaml-cpp
    module unload spectre_gsl
    module unload jemalloc
    module unload libxsmm
    module unload hdf5
    module unload libsharp
    module unload catch
    module unload brigand
    module unload blaze

    spectre_unload_sys_modules
}

spectre_load_modules() {
    spectre_load_sys_modules

    module load blaze
    module load brigand
    module load catch
    module load libsharp
    module load hdf5
    module load libxsmm
    module load jemalloc
    module load spectre_gsl
    module load yaml-cpp
    module load scotch
    module load charm
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    # note that without the below gcc LFS flags, the PCH becomes
    # inconsistent with the source flags, resulting in an invalid PCH
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D DEBUG_SYMBOLS=off \
          -D BUILD_PYTHON_BINDINGS=off \
          -D CMAKE_CXX_FLAGS="-D_FILE_OFFSET_BITS=64 \
-D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE" \
          -D USE_SCOTCH_LB=ON \
          "$@" \
          $SPECTRE_HOME
}
