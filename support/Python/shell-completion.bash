# Distributed under the MIT License.
# See LICENSE.txt for details.

# Generated by click:
# https://click.palletsprojects.com/shell-completion/

_spectre_completion() {
    local IFS=$'\n'
    local response

    response=$(env COMP_WORDS="${COMP_WORDS[*]}" COMP_CWORD=$COMP_CWORD \
_SPECTRE_COMPLETE=bash_complete $1)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"

        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done

    return 0
}

_spectre_completion_setup() {
    complete -o nosort -F _spectre_completion spectre
}

_spectre_completion_setup;

