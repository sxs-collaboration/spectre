#!/bin/bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find all python files in src, support, tests, and tools, then run
# them through yapf.
if ! find ./src/ \
     ./support/ \
     ./tests/ \
     ./tools/ \
     -type f \
     -name "*.py" \
     -print0 \
        | xargs -0 yapf --parallel -d
then
    printf '%s\n' '' '' \
           'Found bad formatting in python files. See diff above' \
           'for details. You can run yapf on a file as:' \
           '  yapf -i PYTHON_FILE.py' \
           'or on the entire repository by running:' \
           '  ./tools/FormatPythonCode.sh' \
           'in the SpECTRE root directory.'
    exit 1
fi

printf "Python formatting is okay."
