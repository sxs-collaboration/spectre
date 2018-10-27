\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing Python Bindings {#spectre_writing_python_bindings}

## CMake and Directory Layout

To allow users to analyze output from simulations and take advantage of
SpECTRE's data structures and functions in python, bindings must sometimes be
written. SpECTRE uses
[Boost.Python]
(https://www.boost.org/doc/libs/1_68_0/libs/python/doc/html/index.html)
to aid with generating the bindings. The C++ code for the bindings should
generally go in a `Python` subdirectory. For example, the bindings for the
DataStructures library would go in `src/DataStructures/Python/`. SpECTRE
provides the `spectre_python_add_module` CMake function to make adding a new
python module, be it with or without bindings, easy.  The python bindings are
built only if `-D BUILD_PYTHON_BINDINGS=ON` is passed when invoking cmake.

The function `spectre_python_add_module` takes as its first argument the module,
in our case
`DataStructures`. The bindings library will be prefixed with `Py`, and therefore
be called `PyDataStructures`. Optionally, a list of `SOURCES` can be passed to
the CMake function. If the python module will only consist of python files, then
the `SOURCES` option should not be specified. Python files that should be part
of the module can be passed with the keyword `PYTHON_FILES`, e.g.
`PYTHON_FILES Hello.py HelloWorld.py`. Finally, the `MODULE_PATH` named argument
can be passed with a string that is the path to where the module should be. For
example, `MODULE_PATH "submodule0/submodule1/"` would mean the module is
accessed from python using `import spectre.submodule0.submodule1.MODULE_NAME`.

Here is a complete example of how to call the `spectre_python_add_module`
function:

\code
spectre_python_add_module(
  ExtraDataStructures
  MODULE_PATH "DataStructures/"
  SOURCES Bindings.cpp MyCoolDataStructure.cpp
  PYTHON_FILES CoolPythonDataStructure.py
  )
\endcode

The library that is added has the name `PyExtraDataStructures` and requires that
at least the `${SPECTRE_LINK_PYBINDINGS}` be set as the last entry to the
`target_link_libraries` call. For example,

\code
target_link_libraries(
  PyExtraDataStructures
  PUBLIC ExtraDataStructures
  ${SPECTRE_LINK_PYBINDINGS}
  )
\endcode

\warning `${SPECTRE_LINK_PYBINDINGS}` must be the *last* argument to
`target_link_libraries`.

## Writing Bindings

Once a python module has been added you can write the actual bindings. You
should structure your bindings directory to reflect the structure of the library
you're writing bindings for. For example, say we want bindings for `DataVector`
and `Matrix` then we should have one source file for each class's bindings
inside `src/DataStructures/Python`. The functions that generate the bindings
should be in the `py_bindings` namespace and have a reasonable name such as
`bind_datavector`. There should be a file named `Bindings.cpp` which calls all
the `bind_*` functions. The `Bindings.cpp` file is quite simple and should
`include <boost/python.hpp>`, forward declare the `bind_*` functions, and then
have `BOOST_PYTHON_MODULE` function. For example,

\code{.cpp}
#include <boost/python.hpp>

namespace py_bindings {
void bind_datavector();
}  // namespace py_bindings

BOOST_PYTHON_MODULE(_DataStructures) {
  Py_Initialize();
  py_bindings::bind_datavector();
}
\endcode

Note that the library name is passed to `BOOST_PYTHON_MODULE` and is prefixed
with an underscore. The underscore is important!

The `DataVector` bindings serve as an example with code comments on how to write
bindings for a class. There is also extensive documentation available directly
from [Boost.Python]
(https://www.boost.org/doc/libs/1_68_0/libs/python/doc/html/index.html)
and [this GitHub repository](https://github.com/TNG/boost-python-examples) also
has many helpful examples. SpECTRE currently aims to support both Python 2.7
and Python 3 and as such all bindings must support both.

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

Below is an example of registering a python test file

\code
spectre_add_python_test(
  "Unit.DataStructures.Python.DataVector"
  Test_DataVector.py
  "unit;datastructures;python")
\endcode

## Using The Bindings

See \ref spectre_using_python "Using SpECTRE's Python"

## Notes:

- All python libraries are dynamic/shared libraries.
- Exceptions should be allowed to propagate through the bindings so that
  error handling via exceptions is possible from python rather than having the
  python interpreter being killed with a call to `abort`.
