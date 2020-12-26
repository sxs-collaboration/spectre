#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

PYTHONPATH="@CMAKE_BINARY_DIR@/bin/python:$PYTHONPATH" @Python_EXECUTABLE@ \
  -m @PYTHON_MODULE_LOCATION@.@PYTHON_FILE_JUST_NAME_WE@
