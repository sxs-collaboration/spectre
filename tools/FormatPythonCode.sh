#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find all python files in src, support, tests, and tools, then run
# them through yapf.
find ./src/ \
     ./support/ \
     ./tests/ \
     ./tools/ \
     -type f \
     -name "*.py" \
     -print0 \
    | xargs -0 yapf --parallel -i
