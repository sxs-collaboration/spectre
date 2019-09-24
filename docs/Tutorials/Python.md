\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Using SpECTRE's Python modules {#spectre_using_python}

Some classes and functions from SpECTRE have Python bindings to make it easier
to visualize data, write test code, and provide an introduction to numerical
relativity without needing to delve into C++.

## Installing the SpECTRE Python modules

First, build SpECTRE with Python bindings enabled by appending the
`-D BUILD_PYTHON_BINDINGS=ON` flag to the `cmake` command. You will find that
a `BUILD_DIR/bin/python` directory is created that contains the Python modules.
Then, you can install the modules into your Python environment in
_development mode_, which means they are symlinked so that changes to the
modules will be reflected in your Python environment, with the command
`pip install -e path/to/bin/python`. Alternatively, remove the `-e` flag to
install the modules normally. You can do this in any Python environment that
supports `pip`, for instance in a
[`virtualenv`/`venv`](https://docs.python.org/3/tutorial/venv.html) or in an
[Anaconda](https://www.anaconda.com/distribution/) environment. You can also get
access to the SpECTRE Python modules by adding  `BUILD_DIR/bin/python` to your
`PYTHONPATH`. This is done automatically by sourcing the `LoadPython.sh` file
with the command `. BUILD_DIR/bin/LoadPython.sh`.
