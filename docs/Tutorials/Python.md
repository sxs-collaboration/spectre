\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Using SpECTRE's Python Modules {#spectre_using_python}

## Prepending to Python Path
Some classes and functions from SpECTRE have Python bindings to make it easier
to visualize data, write test code, and provide an introduction to numerical
relativity without needing to delve into C++. Currently the way to get access to
the SpECTRE python package is to prefix  `BUILD_DIR/bin/python` to your
`PYTHONPATH`. This is done automatically by sourcing the `LoadPython.sh` file by
running `. BUILD_DIR/bin/LoadPython.sh`.
