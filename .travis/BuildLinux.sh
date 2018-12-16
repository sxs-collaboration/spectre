#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Load what we need from `/root/.bashrc`
. /etc/profile.d/lmod.sh
export PATH=$PATH:/work/spack/bin
. /work/spack/share/spack/setup-env.sh
spack load benchmark
spack load blaze
spack load brigand
spack load catch
spack load gsl
spack load libsharp
spack load libxsmm
spack load yaml-cpp
export PATH=$PATH:/work/texlive/bin/x86_64-linux

ccache -z

# We don't need debug symbols during CI, so we turn them off to reduce memory
# usage (by 1.5x) during compilation.
cmake -D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -D CMAKE_C_COMPILER=${CC} \
      -D CMAKE_CXX_COMPILER=${CXX} \
      -D CMAKE_CXX_FLAGS="${CXXFLAGS}" \
      -D CMAKE_Fortran_COMPILER=${FC} \
      -D CHARM_ROOT=/work/charm/multicore-linux64-${CHARM_CC} \
      -D USE_CCACHE=ON \
      -D USE_PCH=${USE_PCH} \
      -D DEBUG_SYMBOLS=OFF \
      -D COVERAGE=${COVERAGE} \
      -D BUILD_PYTHON_BINDINGS=ON \
      ../spectre/

# Build all Charm++ modules
make module_All

if [ ${RUN_CLANG_TIDY} ]; then
    ${SPECTRE_SOURCE_DIR}/.travis/RunClangTidy.sh
fi

if [ ${RUN_IWYU} ]; then
    ${SPECTRE_SOURCE_DIR}/.travis/RunIncludeWhatYouUse.sh
fi

if [ ${BUILD_DOC} ]; then
    # Use `doc-check` target to determine if the documentation was built
    # successfully
    make doc-check || exit 1
fi

# Build the code and run tests
touch ${SPECTRE_BUILD_DIR}/tmp/coverage.info
if [ -z "${RUN_CLANG_TIDY}" ] \
    && [ -z "${RUN_IWYU}" ] \
    && [ -z "${BUILD_DOC}" ]; then
    if [[ ${TRAVIS_BUILD_STAGE_NAME} = "Build libraries" ]]; then
        make libs test-libs-stage1 -j2
    fi

    if [[ ${TRAVIS_BUILD_STAGE_NAME} = \
          "Build and run tests, clangtidy, iwyu, and doxygen" ]]; then
        make -j2
        # Build major executables in serial to avoid hitting memory limits.
        make test-executables -j1
        ctest --output-on-failure -j2

        # Build test coverage
        if [[ ${COVERAGE} = ON ]]; then
            make unit-test-coverage
            if [[ ${TRAVIS_SECURE_ENV_VARS} = true ]]; then
                cd ${SPECTRE_SOURCE_DIR}
                coveralls-lcov -v --repo-token \
                               ${COVERALLS_TOKEN} \
                               ${SPECTRE_BUILD_DIR}/tmp/coverage.info
                cd ${SPECTRE_BUILD_DIR}
            fi
        fi
    fi
    ccache -s
fi

# Build documentation and doc coverage and deploy to GitHub pages.
if [ ${BUILD_DOC} ] \
       && [ ${TRAVIS_SECURE_ENV_VARS} = true ] \
       && [ ${TRAVIS_BRANCH} = ${GH_PAGES_SOURCE_BRANCH} ] \
       && [ ${TRAVIS_PULL_REQUEST} == false ]; then
    make doc-coverage

    # the encrypted key is created using:
    # travis encrypt-file -r sxs-collaboration/spectre deploy_key
    # See:
    # https://gist.github.com/domenic/ec8b0fc8ab45f39403dd
    # https://docs.travis-ci.com/user/encrypting-files/
    openssl aes-256-cbc -K $ENCRYPTED_KEY -iv $ENCRYPTED_IV \
            -in ${SPECTRE_SOURCE_DIR}/.travis/deploy_key.enc \
            -out /root/.ssh/id_rsa -d
    chmod 600 /root/.ssh/id_rsa

    cd /work
    git clone "$GH_PAGES_REPO" gh-pages

    cd /work/gh-pages
    git fetch origin ${GH_PAGES_SOURCE_BRANCH}
    git checkout ${GH_PAGES_SOURCE_BRANCH}
    SHA=`git rev-parse --verify HEAD`

    git checkout --orphan gh-pages

    cp -r /work/build_${BUILD_TYPE}_${CC}/docs/html/* /work/gh-pages/

    git add --all
    git config --global user.name "Automatic Deployment (Travis CI)"
    git config --global user.email "none@none.none"
    git commit -m "Documentation Update: ${SHA}"
    git push -f $GH_PAGES_REPO gh-pages:gh-pages

    rm /root/.ssh/id_rsa*
fi
