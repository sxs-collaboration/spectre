#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# We clear the cache after building dependencies because typically if we had
# to build the dependencies it is because we didn't have a cache to begin
# with. The only case that isn't true is if we upgraded a dependency. Since
# this should be rare we will have a cache when building SpECTRE but not use
# any cache space from ccache of the dependencies.

export CCACHE_COMPRESS=1
# Default compression is 6
export CCACHE_COMPRESSLEVEL=6
export CCACHE_MAXSIZE=5G
export CCACHE_COMPILERCHECK=content

if [ ! -d $DEP_CACHE ]; then
    rm -rf $DEP_CACHE
    mkdir $DEP_CACHE
fi

if [ ! -d $DEP_CACHE/brew ]; then
    rm -rf $DEP_CACHE/brew
    mkdir $DEP_CACHE/brew
fi

# isl, libmpc are GCC dependencies
# szip is an HDF5 dependency
HOMEBREW_DEPS=(mpfr isl libmpc gcc ccache jemalloc gsl szip hdf5 openblas)
HOMEBREW_FORMULA=/usr/local/Homebrew/Library/Taps/homebrew/homebrew-core/Formula

# copy homebrew formulas out of cache to try and avoid having to run
# brew update
FOUND_ALL_DEPS=true
for DEP in ${HOMEBREW_DEPS[*]}
do
    if [ -f $DEP_CACHE/brew/${DEP}.rb ]; then
        cp $DEP_CACHE/brew/${DEP}.rb ${HOMEBREW_FORMULA}
    else
        FOUND_ALL_DEPS=false
    fi
done

# Run brew update if necessary
cd $HOME
rm /usr/local/include/c++
if [ ${FOUND_ALL_DEPS} = false ]; then
    echo '$ brew update'
    brew update
fi

# copy formulas to cache
for DEP in ${HOMEBREW_DEPS[*]}
do
    cp ${HOMEBREW_FORMULA}/${DEP}.rb $DEP_CACHE/brew/
done

# Install dependencies available from homebrew. We explicitly prevent brew from
# automatically updating formulas.
for DEP in ${HOMEBREW_DEPS[*]}
do
    # mpfr is already installed but GCC needs version 4 or newer, so we need
    # to do an upgrade.
    if [ ${DEP} = mpfr ]; then
        echo "$ brew upgrade ${DEP}"
        HOMEBREW_NO_AUTO_UPDATE=1 brew upgrade -f ${DEP}
    else
        echo "$ brew install ${DEP}"
        HOMEBREW_NO_AUTO_UPDATE=1 brew install -f ${DEP}
    fi
done
PATH=/usr/local/opt/ccache/libexec:$PATH

cd $DEP_CACHE

if [ ! -d ./mc ]; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p $DEP_CACHE/mc
    export PATH=$DEP_CACHE/mc/bin:$PATH
    conda info
    conda update --yes conda
    conda create -n osx_env --yes python=$TRAVIS_PYTHON_VERSION
    source activate osx_env
    conda install -y numpy=$NUMPY_VERSION
    conda install -y scipy
    rm ./miniconda.sh
else
    export PATH=$DEP_CACHE/mc/bin:$PATH
    source activate osx_env
fi


if [ ! -d ./libxsmm ]; then
    git clone https://github.com/hfp/libxsmm.git
    cd libxsmm && git checkout 1.8.1
    mkdir build && cd build && make -f ../Makefile
    # clear ccache
    ccache -C
    cd $DEP_CACHE
fi

if [ ! -d ./Catch ]; then
    git clone https://github.com/philsquared/Catch.git
    cd Catch
    git checkout v2.1.2
    cd $DEP_CACHE
fi

if [ ! -d ./brigand ]; then
    git clone https://github.com/edouarda/brigand.git
fi

if [ ! -d ./blaze-${BLAZE_VERSION} ]; then
    wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-${BLAZE_VERSION}.tar.gz
    tar -xzf blaze-${BLAZE_VERSION}.tar.gz
    rm blaze-${BLAZE_VERSION}.tar.gz
fi

if [ "$(sw_vers -productVersion | cut -d '.' -f 1,2)" = "10.12" ]; then
    threading_support=/Library/Developer/CommandLineTools/usr/include/c++/v1/\
__threading_support
    cat >./new_threads <<EOF
#ifndef _MACH_PORT_T
#define _MACH_PORT_T
#include <sys/_types.h> /* __darwin_mach_port_t */
typedef __darwin_mach_port_t mach_port_t;
#include <pthread.h>
mach_port_t pthread_mach_thread_np(pthread_t);
#endif /* _MACH_PORT_T */
EOF
    cat "${threading_support}" >> ./new_threads
    sudo mv ./new_threads "${threading_support}"
fi

if [ ! -d ./charm-${CHARM_VERSION} ]; then
    wget http://charm.cs.illinois.edu/distrib/charm-${CHARM_VERSION}.tar.bz2
    tar -xjf charm-${CHARM_VERSION}.tar.bz2
    cd ./charm-${CHARM_VERSION}
    ./build charm++ multicore-darwin-x86_64 -j2 -O0 $SPECTRE_MIN_MACOS
    git apply $TRAVIS_BUILD_DIR/support/Charm/${CHARM_PATCH}
    # clear ccache
    ccache -C
    rm ../charm-${CHARM_VERSION}.tar.bz2
    cd $DEP_CACHE
fi

if [ ! -d ./yaml-cpp ]; then
    git clone https://github.com/jbeder/yaml-cpp.git
    cd ./yaml-cpp
    mkdir lib
    cd ./lib
    cmake -DBUILD_SHARED_LIBS=ON ../
    make -j2
    # clear ccache
    ccache -C
    cd $DEP_CACHE
fi

cd $HOME
