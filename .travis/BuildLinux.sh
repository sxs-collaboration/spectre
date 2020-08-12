#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

ccache -z

BUILD_PYTHON_BINDINGS=ON
if [ ${BUILD_TYPE} = Release ] && [ ${CC} = clang-8 ]; then
    BUILD_PYTHON_BINDINGS=OFF
fi

# We don't need debug symbols during CI, so we turn them off to reduce memory
# usage (by 1.5x) during compilation.
cmake -D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -D CMAKE_C_COMPILER=${CC} \
      -D CMAKE_CXX_COMPILER=${CXX} \
      -D CMAKE_CXX_FLAGS="${CXXFLAGS}" \
      -D CMAKE_Fortran_COMPILER=${FC} \
      -D CHARM_ROOT=/work/charm_6_10_2/multicore-linux-x86_64-${CHARM_CC} \
      -D USE_CCACHE=ON \
      -D USE_PCH=${USE_PCH} \
      -D DEBUG_SYMBOLS=OFF \
      -D COVERAGE=${COVERAGE} \
      -D BUILD_PYTHON_BINDINGS=${BULID_PYTHON_BINDINGS} \
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
    if [[ ${TRAVIS_BUILD_STAGE_NAME} = "Build stage 1" ]]; then
        time make libs -j2
        time make test-libs-domain -j2
        time make test-libs-elliptic -j2
        time make test-libs-evolution -j2
    fi

    if [[ ${TRAVIS_BUILD_STAGE_NAME} = "Build stage 2" ]]; then
        time make test-libs-numerical-algorithms -j2
        time make test-libs-parallel-algorithms -j2
        # Build DataStructures in serial because the DataVector tests
        # are very memory intensive to compile
        time make test-libs-data-structures -j1
        time make test-libs-pointwise-functions -j2
    fi

    if [[ ${TRAVIS_BUILD_STAGE_NAME} = "Build stage 3" ]]; then
        time make test-libs-other -j2
        time make -j2
    fi

    if [[ ${TRAVIS_BUILD_STAGE_NAME} = \
          "Build stage 4, clangtidy, and doxygen" ]]; then
        # Build major executables in serial to avoid hitting memory limits.
        time make test-executables -j1
        time ctest --output-on-failure -j2

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
# Disabled since this is done with GitHub actions instead of Travis.
DEPLOY_DOC=false
if [ ${BUILD_DOC} ] \
       && [ ${DEPLOY_DOC} = true ] \
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
