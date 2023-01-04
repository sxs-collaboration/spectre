#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

PYTHONPATH="@PYTHONPATH@" @JEMALLOC_PRELOAD@ @Python_EXECUTABLE@ \
  -m @PYTHON_SCRIPT_LOCATION@ "$@"
