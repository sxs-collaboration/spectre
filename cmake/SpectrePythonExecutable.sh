#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

PYTHONPATH="@PYTHONPATH@" @PYTHON_EXEC_ENV_VARS@ @Python_EXECUTABLE@ \
  @PYTHON_EXE_COMMAND@ "$@"
