#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

export PATH=$DEP_CACHE/mc/bin:$PATH
source activate osx_env

# We haven't figured out how to get ccache working on macOS properly, either
# with or without a PCH.
export USE_PCH=ON

# Move the build dir somewhere that is guaranteed not to depend on the
# repository being built, the branch, or anything else. ${HOME} on macOS
# will be something (reasonably) stable and has been /Users/travis for the
# last several years now.
mv ${TRAVIS_BUILD_DIR} ${HOME}

export SPECTRE_BUILD_DIR=${HOME}/build
export SPECTRE_SOURCE_DIR=${HOME}/spectre

# export CCACHE_LOGFILE=$HOME/ccache_log.txt
export CCACHE_COMPRESS=1
# Default compression is 6
export CCACHE_COMPRESSLEVEL=6
export CCACHE_MAXSIZE=5G
export CCACHE_COMPILERCHECK=none
export CCACHE_EXTRAFILES="${SPECTRE_SOURCE_DIR}/tools/SpectrePch.hpp"
export CCACHE_IGNOREHEADERS="${SPECTRE_BUILD_DIR}/SpectrePch.hpp:\
${SPECTRE_BUILD_DIR}/SpectrePch.hpp.gch"

# Print the ccache setting. Useful for finding ccache-related bugs.
ccache -p

if [ ! -d $SPECTRE_BUILD_DIR ]; then
    rm -rf $SPECTRE_BUILD_DIR
    mkdir $SPECTRE_BUILD_DIR
fi
cd $SPECTRE_BUILD_DIR
ccache -z

# We don't need debug symbols during CI, so we turn them off to reduce memory
# usage (by 1.5x) during compilation.
cmake -D CHARM_ROOT=$DEP_CACHE/charm-${CHARM_VERSION} \
      -D BLAZE_ROOT=$DEP_CACHE/blaze-${BLAZE_VERSION}/ \
      -D BRIGAND_ROOT=$DEP_CACHE/brigand\
      -D CATCH_ROOT=$DEP_CACHE/Catch\
      -D LIBXSMM_ROOT=$DEP_CACHE/libxsmm/build\
      -D YAMLCPP_ROOT=$DEP_CACHE/yaml-cpp/\
      -D MACOSX_MIN=10.11\
      -D BLAS_openblas_LIBRARY=/usr/local/opt/openblas/lib/libopenblas.dylib\
      -D LAPACK_openblas_LIBRARY=/usr/local/opt/openblas/lib/libopenblas.dylib\
      -D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -D USE_PCH=${USE_PCH} \
      -D DEBUG_SYMBOLS=OFF \
      ${SPECTRE_SOURCE_DIR}

# Build all Charm++ modules
make module_All

if [[ ${TRAVIS_BUILD_STAGE_NAME} = "Build libraries" ]]; then
    make libs test-libs-stage1 -j2
fi

if [[ ${TRAVIS_BUILD_STAGE_NAME} = \
      "Build and run tests, clangtidy, iwyu, and doxygen" ]]; then
    make test-executables \
         $([ $BUILD_TYPE = Release ] && echo Benchmark) -j2
    ctest --output-on-failure -j2
fi

ccache -s;
