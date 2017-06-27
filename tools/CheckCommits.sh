#!/usr/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Set \n as the only separator for lists since commit messages might
# have spaces in them
IFS=$'\n'

# Get the last 300 commits and check those for bad messages. TravisCI doesn't
# allow us to get develop (at least not easily) so we can't just find and check
# the added commits.
COMMIT_LINES=`git log -n 300 --oneline`

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
