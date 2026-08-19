#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find the Python package next to this script's location, so that we can copy
# this script to bin directories (see `support/Python/BinDirectory.py`)
SPECTRE_BIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHONPATH="${SPECTRE_BIN_DIR}/python:@PYTHONPATH@" @PYTHON_EXEC_ENV_VARS@ \
  @Python_EXECUTABLE@ @PYTHON_EXE_COMMAND@ "$@"
