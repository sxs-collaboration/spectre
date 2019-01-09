#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Setup lmod and spack to load dependencies
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

BUILD_DIR=`pwd`
git clone ${UPSTREAM_REPO} /work/spectre_upstream
cd /work/spectre_upstream
git checkout ${UPSTREAM_BRANCH}
COMMITS_ON_UPSTREAM=`git rev-list HEAD`
cd /work/spectre

# For each upstream commit we check if the commit is on this branch, once we
# find a match we save that hash and exit. This allows us to check only files
# currently being committed.
UPSTREAM_COMMIT_HASH=''

for HASH in ${COMMITS_ON_UPSTREAM}
do
    if git cat-file -e $HASH^{commit} 2> /dev/null
    then
       UPSTREAM_COMMIT_HASH=$HASH
       break
    fi
done

if [ -z $UPSTREAM_COMMIT_HASH ];
then
   echo "The branch is not branched from ${UPSTREAM_REPO}/${UPSTREAM_BRANCH}"
   exit 1
fi

echo "Using upstream commit hash: ${UPSTREAM_COMMIT_HASH}"

###############################################################################
# Get list of non-deleted files
MODIFIED_FILES=''

for FILENAME in `git diff --name-only ${UPSTREAM_COMMIT_HASH} HEAD`
do
    if [ -f $FILENAME ] \
           && [ ${FILENAME: -4} == ".cpp" ] \
           && ! grep -q "FILE_IS_COMPILATION_TEST" $FILENAME; then
        MODIFIED_FILES="${MODIFIED_FILES} $FILENAME"
    fi
done

# Go to build directory and then run clang-tidy, writing output to file
cd ${BUILD_DIR}
CLANG_TIDY_OUTPUT="./.clang_tidy_check_output"
rm -f ${CLANG_TIDY_OUTPUT}

printf "\nRunning clang-tidy on:\n"
echo $MODIFIED_FILES
echo ''

for FILENAME in $MODIFIED_FILES
do
    # need to output something so TravisCI knows we're not stalled
    printf '.'
    printf "\n\nRunning clang-tidy on file $FILENAME\n" >> ${CLANG_TIDY_OUTPUT}
    make clang-tidy FILE=/work/spectre/${FILENAME} >> ${CLANG_TIDY_OUTPUT} 2>&1
done
echo ''

if [ -f "${CLANG_TIDY_OUTPUT}" ]; then
    sed -i'.bak' "s^warning: /usr/bin/clang++: 'linker' input unused.*^^g" \
        ${CLANG_TIDY_OUTPUT}
    sed -i'.bak' "s^warning: ccache: 'linker' input unused.*^^g" \
        ${CLANG_TIDY_OUTPUT}
fi

if [ -f "${CLANG_TIDY_OUTPUT}" ] \
       && ( grep 'warning:' ${CLANG_TIDY_OUTPUT} > /dev/null ||
                grep 'error:' ${CLANG_TIDY_OUTPUT} > /dev/null )
then
    sed -i '/.*Built target.*/d'  ${CLANG_TIDY_OUTPUT}
    sed -i '/.*Generating .*/d'  ${CLANG_TIDY_OUTPUT}
    printf "\nclang-tidy found problems! Output:\n\n"
    more ${CLANG_TIDY_OUTPUT}
    exit 1
fi

exit 0
