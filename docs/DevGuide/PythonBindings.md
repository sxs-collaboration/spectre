\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing Python Bindings {#spectre_writing_python_bindings}

## CMake and Directory Layout

To allow users to analyze output from simulations and take advantage of
SpECTRE's data structures and functions in python, bindings must sometimes be
written. SpECTRE uses [pybind11](https://pybind11.readthedocs.io/)
to aid with generating the bindings. The C++ code for the bindings should
generally go in a `Python` subdirectory. For example, the bindings for the
DataStructures library would go in `src/DataStructures/Python/`. SpECTRE
provides the `spectre_python_add_module` CMake function to make adding a new
python module, be it with or without bindings, easy.  The python bindings are
built only if `-D BUILD_PYTHON_BINDINGS=ON` is passed when invoking cmake.
You can specify the Python version, interpreter and libraries used for compiling
and testing the bindings by setting the `-D Python_EXECUTABLE` to an absolute
path such as `/usr/bin/python3`.

The function `spectre_python_add_module` takes as its first argument the module,
in our case `DataStructures`. Optionally, a list of `SOURCES` can be passed to
the CMake function. If you specify `SOURCES`, you must also specify a
`LIBRARY_NAME`. A good `LIBRARY_NAME` is the name of the C++ library for which
bindings are being built prefixed with `Py`, e.g. `PyDataStructures`. If the
Python module will only consist of Python files, then the `SOURCES` option
should not be specified. Python files that should be part
of the module can be passed with the keyword `PYTHON_FILES`, e.g.
`PYTHON_FILES Hello.py HelloWorld.py`. Finally, the `MODULE_PATH` named argument
can be passed with a string that is the path to where the module should be. For
example, `MODULE_PATH "submodule0/submodule1/"` would mean the module is
accessed from python using `import spectre.submodule0.submodule1.MODULE_NAME`.

Here is a complete example of how to call the `spectre_python_add_module`
function:

\code
spectre_python_add_module(
  Extra
  LIBRARY_NAME "PyExtraDataStructures"
  MODULE_PATH "DataStructures/"
  SOURCES Bindings.cpp MyCoolDataStructure.cpp
  PYTHON_FILES CoolPythonDataStructure.py
  )
\endcode

The library that is added has the name `PyExtraDataStructures`. Make sure to
call `spectre_python_link_libraries` for every Python module that compiles
`SOURCES`. For example,

\code
spectre_python_link_libraries(
  PyExtraDataStructures
  PUBLIC ExtraDataStructures
  )
\endcode

You may also call `spectre_python_add_dependencies` for Python modules that
have `SOURCES`, e.g.

\code
spectre_python_add_dependencies(
  PyExtraDataStructures
  PyDataStructures
  )
\endcode

Note that these functions will skip adding or configure any C++ libraries if
the `BUILD_PYTHON_BINDINGS` flag is `OFF`.

## Writing Bindings

Once a python module has been added you can write the actual bindings. You
should structure your bindings directory to reflect the structure of the library
you're writing bindings for. For example, say we want bindings for `DataVector`
and `Matrix` then we should have one source file for each class's bindings
inside `src/DataStructures/Python`. The functions that generate the bindings
should be in the `py_bindings` namespace and have a reasonable name such as
`bind_datavector`. There should be a file named `Bindings.cpp` which calls all
the `bind_*` functions. The `Bindings.cpp` file is quite simple and should
`include <pybind11/pybind11.h>`, forward declare the `bind_*` functions, and
then have a `PYBIND11_MODULE` function. For example,

\code{.cpp}
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace py_bindings {
void bind_datavector(py::module& m);
}  // namespace py_bindings

PYBIND11_MODULE(_PyDataStructures, m) {
  py_bindings::bind_datavector(m);
}
\endcode

Note that the library name is passed to `PYBIND11_MODULE` and is prefixed
with an underscore. The underscore is important and the library name must be the
same that is passed as `LIBRARY_NAME` to `spectre_python_add_module` (see
above).

The `DataVector` bindings serve as an example with code comments on how to write
bindings for a class. There is also extensive documentation available directly
from [pybind11](https://pybind11.readthedocs.io/). SpECTRE currently aims to
support both Python 2.7 and Python 3 and as such all bindings must support both.

\note Exceptions should be allowed to propagate through the bindings so that
error handling via exceptions is possible from python rather than having the
python interpreter being killed with a call to `abort`.

## Testing Python Bindings and Code

All the python bindings must be tested. SpECTRE uses the
[unittest](https://docs.python.org/3/library/unittest.html) framework
provided as part of python. To register a test file with CMake use the
SpECTRE-provided function `spectre_add_python_test` passing as the first
argument the test name (e.g. `"Unit.DataStructures.Python.DataVector"`), the
file as the second argument (e.g. `Test_DataVector.py`), and a semicolon
separated list of labels as the last (e.g. `"unit;datastructures;python"`).
All the test cases should be in a single class so that the python unit testing
framework will run all test functions on a single invocation to avoid startup
cost.

Below is an example of registering a python test file for bindings:

\snippet tests/Unit/DataStructures/CMakeLists.txt example_add_pybindings_test

Python code that does not use bindings must also be tested. You can register the
test file using the `spectre_add_python_test` CMake function with the same
signature as shown above.

Please note that the tests must be formatted according to the `.style.yapf` file
in the root of the repository.

## Using The Bindings

See \ref spectre_using_python "Using SpECTRE's Python"

## Notes:

- All python libraries are dynamic/shared libraries.
- Exceptions should be allowed to propagate through the bindings so that
  error handling via exceptions is possible from python rather than having the
  python interpreter being killed with a call to `abort`.
- All function arguments in Python bindings should be named using `py::arg`.
  See the Python bindings in `IO/H5/` for examples. Using the named arguments in
  Python code is optional, but preferred when it makes code more readable.
  In particular, use the argument names in the tests for the Python bindings so
  they are being tested as well.
