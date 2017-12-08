#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

exec @LLVM_COV_BIN@ gcov "$@"
