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
    printf "\n\nFound bad formatting in python files. See diff above \n" \
           "for details. You can run yapf on a file as:\n" \
           "  yapf -i PYTHON_FILE.py\n" \
           "or on the entire repository by running:\n" \
           "  ./tools/FormatPythonCode.py\n" \
           "in the SpECTRE root directory.\n"
    exit 1
fi

printf "Python formatting is okay."
