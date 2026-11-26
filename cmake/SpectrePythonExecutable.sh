#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find the Python package next to this script, so that a copy of the script in
# a simulation's bin directory uses the package next to the copy
SPECTRE_BIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHONPATH="${SPECTRE_BIN_DIR}/python:@PYTHONPATH@" @PYTHON_EXEC_ENV_VARS@ \
  @Python_EXECUTABLE@ @PYTHON_EXE_COMMAND@ "$@"
