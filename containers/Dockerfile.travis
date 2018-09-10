# Distributed under the MIT License.
# See LICENSE.txt for details.

# We derive from the SpECTRE build environment Docker image which contains all
# the dependencies that are needed to compile SpECTRE. Deriving off the build
# environment Docker image is substantially faster than installing compilers
# and other dependencies each time we run a build.
#
# The spectrebuildenv Docker image is generated using the Dockerfile.buildenv
# by running (as root):
# docker build  -t sxscollaboration/spectrebuildenv:latest \
#               -f ./spectre/containers/Dockerfile.buildenv .
#
# For those who are maintaining the build env, see that Dockerfile for
# instructions on pushing to the SXS collaboration DockerHub account.
FROM sxscollaboration/spectrebuildenv:latest

# All ARG variables are passed into Docker as a
# `--build-arg VARIABLE_NAME=VARIABLE_VALUE`
ARG COVERAGE=OFF
ARG CC
ARG CXX
ARG FC
ARG CHARM_CC
ARG CXXFLAGS
ARG BUILD_TYPE
ARG DOCUMENTATION_ONLY

# Environment variables used for test and documentation code coverage analysis
ARG TRAVIS_BRANCH
ARG TRAVIS_JOB_NUMBER
ARG TRAVIS_PULL_REQUEST
ARG TRAVIS_JOB_ID
ARG TRAVIS_TAG
ARG TRAVIS_REPO_SLUG
ARG TRAVIS_COMMIT
ARG TRAVIS_EVENT_TYPE
ARG TRAVIS_BUILD_STAGE_NAME
ARG TRAVIS_SECURE_ENV_VARS
ARG ENCRYPTED_KEY
ARG ENCRYPTED_IV
ARG GH_PAGES_REPO
ARG GH_PAGES_SOURCE_BRANCH
ARG COVERALLS_TOKEN
ARG RUN_CLANG_TIDY
ARG RUN_IWYU
ARG BUILD_DOC
ARG USE_PCH
ARG UPSTREAM_REPO
ARG UPSTREAM_BRANCH
ARG CHARM_PATCH
ARG CORE_LIBS

ENV SPECTRE_BUILD_DIR=/work/build_${BUILD_TYPE}_${CC}
ENV SPECTRE_SOURCE_DIR=/work/spectre

# In order to have ccache work on TravisCI with a precompiled header we need to:
# - Hash the content of the compiler rather than the location and mtime
# - Hash the header file in the repo that will generate the precompiled header
# - Ignore the precompiled header in the build directory
# ENV CCACHE_LOGFILE=/work/ccache_log.txt
ENV CCACHE_COMPRESS=1
# Default compression is 6
ENV CCACHE_COMPRESSLEVEL=6
ENV CCACHE_MAXSIZE=5G
ENV CCACHE_COMPILERCHECK=content
ENV CCACHE_EXTRAFILES="${SPECTRE_SOURCE_DIR}/tools/SpectrePch.hpp"
ENV CCACHE_IGNOREHEADERS="${SPECTRE_BUILD_DIR}/SpectrePch.hpp:${SPECTRE_BUILD_DIR}/SpectrePch.hpp.gch"

# Print the ccache configuration out
RUN ccache -p

# Copy the spectre repo from the TravisCI disk into the Docker image /work
# directory
COPY spectre ${SPECTRE_SOURCE_DIR}

# We need the GitHub host added to known_hosts. It is safer to copy in the
# known_hosts from Travis than to add the REPO host automatically, since that
# could be changed to point anywhere.
RUN mkdir /root/.ssh
COPY known_hosts /root/.ssh/known_hosts

# Copy ccache from TravisCI cached data (we cache the ccache cache between
# builds) to the Docker image
COPY ccache /root/.ccache

# Create build directory
RUN mkdir ${SPECTRE_BUILD_DIR}
WORKDIR ${SPECTRE_BUILD_DIR}

# Run the main build script
RUN ${SPECTRE_SOURCE_DIR}/.travis/BuildLinux.sh;
