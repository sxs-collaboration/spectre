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
`-D BUILD_PYTHON_BINDINGS=ON` flag to the `cmake` command. You can specify the
Python version, interpreter and libraries used for compiling and testing the
bindings by setting the `-D Python_EXECUTABLE` to an absolute path such as
`/usr/bin/python3`. You will find that
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

## Running Jupyter within the Docker container

[Jupyter lab](https://jupyterlab.readthedocs.io/) is installed in the Docker
container. You can run it in the container and access it through a browser on
the host for a convenient way to work with the SpECTRE Python bindings. To do
so, make sure you have exposed a port when running the Docker container, e.g.
by appending the option `-p 8000:8000` to the `docker run` command (see
\ref installation). Inside the docker container, it can be convenient to
use `disown` or to `apt-get install screen` and use `screen` to obtain a shell
that runs the Jupyter server permanently in the background. Within the shell you
want to run your Jupyter server, navigate to a directory that will serve as the
root for the file system that Jupyter has access to. Make sure it is shared with
the host (e.g. `SPECTRE_HOME`) so your Jupyter notebooks are not lost when the
container is deleted. Then, run the following command:

```sh
jupyter lab --ip 0.0.0.0 --port 8000 --allow-root
```

Copy the token that is being displayed. Now you can open a browser on the host
machine, point it to `http://localhost:8000` and paste in the token. You will
have access to the Python environment within the Docker container. If you have
followed the instructions above for installing the SpECTRE Python package,
you can try importing the Python package in a notebook with:

```py
import spectre
```

While developing Python code in the `spectre` packages it can be useful to
configure Jupyter to reload packages when they change. Add the following code
before any import statements:

```
%load_ext autoreload
%autoreload 2
```
