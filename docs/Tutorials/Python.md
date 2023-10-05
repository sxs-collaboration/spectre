\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Using SpECTRE's Python modules {#spectre_using_python}

\tableofcontents

Some classes and functions from SpECTRE have Python bindings to make it easier
to visualize data, write test code, and provide an introduction to numerical
relativity without needing to delve into C++.

## Building the SpECTRE Python modules

> tl;dr: Compile the `all-pybindings` or `cli` target.

First, build SpECTRE with Python bindings enabled by appending the
`-D BUILD_PYTHON_BINDINGS=ON` flag to the `cmake` command (enabled by default).
You can specify the
Python version, interpreter and libraries used for compiling and testing the
bindings by setting the `-D Python_EXECUTABLE` to an absolute path such as
`/usr/bin/python3`. Compile the `all-pybindings` (or `cli`) target. You will
find that a `BUILD_DIR/bin/python` directory is created that contains the
Python modules.

## Importing the SpECTRE Python modules

> tl;dr: Spin up a Jupyter notebook with `./bin/python-spectre -m jupyterlab`,
> then `import spectre`.

You have many options for making the Python modules accessible to import in your
scripts:

- Use the `BUILD_DIR/bin/python-spectre` shortcut to run your scripts. It just
  points the `PYTHONPATH` to the modules in `BUILD_DIR/bin/python` and invokes
  the Python interpreter that you configured your build with. You can also use
  this shortcut to spin up a Jupyter server:

  ```sh
  BUILD_DIR/bin/python-spectre -m jupyterlab
  ```

  If Jupyter is not yet installed in your Python environment, you can install it
  like this:

  ```sh
  BUILD_DIR/bin/python-spectre -m pip install jupyterlab
  ```

  > Note for VSCode users: You can also select this Jupyter server as kernel for
  > notebooks running in VSCode (see [docs](https://code.visualstudio.com/docs/datascience/jupyter-notebooks#_connect-to-a-remote-jupyter-server)).

- Install the modules into your Python environment:

  ```sh
  pip install [-e] BUILD_DIR/bin/python
  ```

  The optional `-e` flag installs the modules in _development mode_, which means
  they are symlinked so that changes to the modules will be reflected in your
  Python environment.

  You can install the Python modules like this in any Python environment that
  supports `pip`, for instance in a
  [`virtualenv`/`venv`](https://docs.python.org/3/tutorial/venv.html) or in an
  [Anaconda](https://www.anaconda.com/distribution/) environment.

- You can also get access to the SpECTRE Python modules by manually adding
  `BUILD_DIR/bin/python` to your `PYTHONPATH`. This is done automatically by
  the `LoadPython.sh` script:

  ```sh
  . BUILD_DIR/bin/LoadPython.sh
  ```

  Note that by default SpECTRE uses `jemalloc` which needs to be pre-loaded for
  the Python bindings to work. Therefore, you need to run
  `LD_PRELOAD=/path/to/libjemalloc.so python` to execute Python scripts or start
  Python consoles. The path to your preferred jemalloc installation is printed
  out at the end of the `cmake` configuration or can be found by running the
  script `BUILD_DIR/bin/LoadPython.sh`. Alternatively, you can use your system's
  memory allocator by appending the flag `-D MEMORY_ALLOCATOR=SYSTEM` to the
  `cmake` command. In this case you will not need to pre-load any libraries.

Using any of the above options you should be able to import the `spectre` Python
modules in your scripts or notebooks. You can try it like this:

```py
import spectre
```

For an overview of all available Python modules see the
[Python modules documentation](py/_autosummary/spectre.html). For an example
what you can do with them see the tutorial on \ref tutorial_vis_python.

## Running Jupyter within the Docker container

You can run [Jupyter lab](https://jupyterlab.readthedocs.io/) in the Docker
container and access it through a browser on
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
BUILD_DIR/bin/python-spectre -m jupyterlab --ip 0.0.0.0 --port 8000 --allow-root
```

Copy the token that is being displayed. Now you can open a browser on the host
machine, point it to `http://localhost:8000` and paste in the token. You will
have access to the Python environment within the Docker container.

## Developing Python code

> tl;dr: Set `PY_DEV_MODE=ON` in your CMake configuration so Python files are
> symlinked to the build directory.

When you edit any of the Python files in the repository, the changes are not
immediately reflected in the Python modules that you import from the build
directory. You have to run `cmake .` in the build directory to update the Python
files. This can be avoided by setting `cmake -D PY_DEV_MODE=ON .` in the build
directory, which configures CMake to symlink the Python files rather than copy
them. Some features that rely on CMake variables break with this mode, e.g. the
SpECTRE version number is not inserted into the Python modules so
`spectre.__version__` will return nonsense. Nonetheless, enabling this mode
avoids a lot of the hassle that comes from re-configuring CMake every time you
change a file.

Furthermore, enabling `autoreload` in Jupyter notebooks can be very helpful when
you are editing Python code. Add the following code before any import statements
in your notebook:

```
%load_ext autoreload
%autoreload 2
```

For details on writing Python bindings for our C++ functions, rather than
writing Python code, see the developer guide on
\ref spectre_writing_python_bindings.
