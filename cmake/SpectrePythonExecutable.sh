#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

PYTHONPATH="@PYTHONPATH@" @Python_EXECUTABLE@ \
  -m @PYTHON_SCRIPT_LOCATION@ "$@"
