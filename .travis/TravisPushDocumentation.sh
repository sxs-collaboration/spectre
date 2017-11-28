#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# This script is called from Dockerfile.travis, which is where the various
# environment variables are defined, or received from TravisCI.

# Add executables for SonarSQUBE to PATH
export PATH=$PATH:/work/sonarcloud/sonar-scanner-2.8/bin
export PATH=$PATH:/work/sonarcloud/build-wrapper-linux-x86

# Setup lmod and spack to load dependencies
. /etc/profile.d/lmod.sh
export PATH=$PATH:/work/spack/bin
. /work/spack/share/spack/setup-env.sh
spack load blaze
spack load brigand
spack load catch
spack load libxsmm
spack load yaml-cpp

# We use cron jobs to deploy to gh-pages. Since this still runs all jobs we
# only actually build documentation for one job but let the others run tests.
if [ ${CC} = gcc ] \
&& [ ${TRAVIS_SECURE_ENV_VARS} = true ] \
&& [ ${COVERAGE} ] \
&& [ ${TRAVIS_BRANCH} = ${GH_PAGES_SOURCE_BRANCH} ] \
&& [ ${TRAVIS_PULL_REQUEST} == false ]; then
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
