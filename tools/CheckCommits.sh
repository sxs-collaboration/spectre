#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Set \n as the only separator for lists since commit messages might
# have spaces in them
IFS=$'\n'

# We get the list of commit messages to check. We only check commits that are
# not on the develop branch. For non-Travis runs, from a user fork we assume
# that the branch 'develop' exists and only check the commits since the HEAD
# of the develop branch
# On TravisCI we clone the upstream repository and use the UPSTREAM_BRANCH
# (typically 'develop') to check for each upstream commit if it is on the
# pull request branch, once we find a match we save that hash and
# exit. This allows us to check only files currently being committed.
# Note that the search is in reverse chronological order.
COMMIT_LINES=''
if [ -z $TRAVIS ];
then
    COMMIT_LINES=`git log develop..HEAD --oneline`
else
    WORK_DIR=`pwd`
    git clone ${UPSTREAM_REPO} ${HOME}/spectre_upstream
    cd ${HOME}/spectre_upstream
    git checkout ${UPSTREAM_BRANCH}
    COMMITS_ON_UPSTREAM=`git rev-list HEAD`
    cd $WORK_DIR
    # For each upstream commit we check if the commit is on this
    # branch, once we find a match we save that hash and exit. This
    # allows us to check only files currently being committed.
    HEAD_COMMIT_HASH=''

    for hash in ${COMMITS_ON_UPSTREAM}
    do
        if git cat-file -e $hash^{commit} 2> /dev/null
        then
            HEAD_COMMIT_HASH=$hash
            break
        fi
    done

    if [ -z $HEAD_COMMIT_HASH ];
    then
        echo "The branch is not branched from" \
             "${UPSTREAM_REPO}/${UPSTREAM_BRANCH}"
        exit 1
    fi
    COMMIT_LINES=`git log ${HEAD_COMMIT_HASH}..HEAD --oneline`
fi

# For all commit messages, check if they start with one of the key words
for commit_msg in $COMMIT_LINES
do
    if grep "^[0-9a-f]\{6,40\} [Ff][Ii][Xx][Uu][Pp]" <<< $commit_msg\
    > /dev/null; then
        printf "\n\n\nError: Fixup commit found: %s\n\n\n" "$commit_msg" 2>&1
        exit 1
    elif grep "^[0-9a-f]\{6,40\} [Ww][Ii][Pp]" <<< $commit_msg > /dev/null; then
        printf "\n\n\nError: WIP commit found: %s\n\n\n" "$commit_msg" 2>&1
        exit 1
    elif grep "^[0-9a-f]\{6,40\} [Ff][Ii][Xx][Mm][Ee]" <<< $commit_msg\
    > /dev/null; then
        printf "\n\n\nError: FixMe commit found: %s\n\n\n" "$commit_msg" 2>&1
        exit 1
    elif grep "^[0-9a-f]\{6,40\} [Dd][Ee][Ll][Ee][Tt][Ee][Mm][Ee]" <<< \
    $commit_msg > /dev/null; then
        printf "\n\n\nError: DeleteMe commit found: %s\n\n\n" "$commit_msg" 2>&1
        exit 1
    elif grep "^[0-9a-f]\{6,40\} [Rr][Ee][Bb][Aa][Ss][Ee][Mm][Ee]" <<< \
    $commit_msg > /dev/null; then
        printf "\n\n\nError: RebaseMe commit found: %s\n\n\n" "$commit_msg" 2>&1
        exit 1
    elif grep "^[0-9a-f]\{6,40\} [Tt][Ee][Ss][Tt][Ii][Nn][Gg]" <<< $commit_msg\
     > /dev/null; then
        printf "\n\n\nError: Testing commit found: %s\n\n\n" "$commit_msg" 2>&1
        exit 1
    elif grep "^[0-9a-f]\{6,40\} [Rr][Ee][Bb][Aa][Ss][Ee]" <<< $commit_msg\
     > /dev/null; then
        printf "\n\n\nError: Rebase commit found: %s\n\n\n" "$commit_msg" 2>&1
        exit 1
    fi
done

exit 0
