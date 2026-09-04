#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    echo "All modules on Mbot are provided by the system"
}

spectre_load_modules() {
    export SPECTRE_MACHINE=Mbot
    # The order here is important
    module load gcc/11.4.0
    module load spectre-deps > /dev/null 2>&1
}

spectre_unload_modules() {
    # The order here is important
    module unload spectre-deps > /dev/null 2>&1
    module unload gcc/11.4.0
}

spectre_run_cmake_gcc() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules > /dev/null 2>&1
    cmake -S $SPECTRE_HOME -B . --preset release-debug "$@"
}

spectre_run_cmake_clang() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules > /dev/null 2>&1
    cmake -S $SPECTRE_HOME -B . --preset release-debug-clang "$@"
}
