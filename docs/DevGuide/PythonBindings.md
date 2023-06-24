\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing Python Bindings {#spectre_writing_python_bindings}

\tableofcontents

## CMake and Directory Layout

To allow users to analyze output from simulations and take advantage of
SpECTRE's data structures and functions in python, bindings must sometimes be
written. SpECTRE uses [pybind11](https://pybind11.readthedocs.io/)
to aid with generating the bindings. The C++ code for the bindings should
generally go in a `Python` subdirectory. For example, the bindings for the
DataStructures library would go in `src/DataStructures/Python/`. SpECTRE
provides the `spectre_python_add_module` CMake function to make adding a new
python module, be it with or without bindings, easy.  The python bindings are
built only if `-D BUILD_PYTHON_BINDINGS=ON` is passed when invoking cmake
(enabled by default).
You can specify the Python version, interpreter and libraries used for compiling
and testing the bindings by setting the `-D Python_EXECUTABLE` to an absolute
path such as `/usr/bin/python3`.

The function `spectre_python_add_module` takes as its first argument the module,
in our case `DataStructures`. Optionally, a list of `SOURCES` can be passed to
the CMake function. If you specify `SOURCES`, you must also specify a
`LIBRARY_NAME`. A good `LIBRARY_NAME` is the name of the C++ library for which
bindings are being built prefixed with `Py`, e.g. `PyDataStructures`. If the
Python module will only consist of Python files, then the `SOURCES` option
should not be specified. Python files that should be part of the module can be
passed with the keyword `PYTHON_FILES`. Finally, the `MODULE_PATH`
named argument can be passed with a string that is the path to where the module
should be. For example, `MODULE_PATH "submodule0/submodule1/"` would mean the
module is accessed from python using
`import spectre.submodule0.submodule1.MODULE_NAME`.

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
  PRIVATE
  ExtraDataStructures
  pybind11::module
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

PYBIND11_MODULE(_Pybindings, m) {
  py_bindings::bind_datavector(m);
}
\endcode

Note that the library name is passed to `PYBIND11_MODULE` and is prefixed
with an underscore. The underscore is important and the library name must be the
same that is passed as `LIBRARY_NAME` to `spectre_python_add_module` (see
above).

The `DataVector` bindings serve as an example with code comments on how to write
bindings for a class. There is also extensive documentation available directly
from [pybind11](https://pybind11.readthedocs.io/).

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

## Guidelines for writing command-line interfaces (CLIs)

- List all CLI endpoints in `support/Python/__main__.py`.
- Follow the recommendations in the
  [click](https://click.palletsprojects.com/en/8.1.x/) documentation.
- Split your code into free functions that know nothing about the CLI and can
  just as well be called independently from Python, and the CLI commands that
  call the functions. Test both.
- Take only input files that the script operates on as positional arguments
  (like H5 data files or YAML input files) and everything else as options.
- Choose option names and shorthands consistent with other CLI endpoints in the
  repository. For example, H5 subfile names are specified with '--subfile-name'
  / '-d' and output files are specified with '--output' / '-o'. Look at other
  CLI endpoints before making choices for option names.
- Never read or write files to or from "default" locations. Instead, take all
  input files as arguments and write all output files to locations specified
  explicitly by the user. This is important so users are not afraid of moving
  and renaming files, and are not left wondering where the script wrote its
  output. Examples:
  - Don't try to read a file like "spectre.out" from the current directory just
    because it might be there by convention. Instead, add an argument or option
    like `--out-filename` so the user can specify it.
  - Don't write a file like "plot.pdf" to the current directory without telling
    the user. Instead, add an option like `--output` / `-o` for the user to
    specify explicitly so they know exactly where output is written to.
- Operate on files instead of directories when possible. For example, prefer
  taking many H5 volume data files as arguments instead of the directory that
  contains them. This helps with operating on H5 files in segments or other
  subdirectory structures. Passing many files to a script is easy for the user
  by using a glob (note: don't take the glob as a string argument, take the
  expanded list of files directly using `click.argument(..., nargs=-1,
  type=click.Path(...))`).
- Never overwrite or delete files without prompting the user or asking them to
  run with `--force`.
- When the input to a script is empty, [gracefully degrade to a
  noop](https://click.palletsprojects.com/en/8.1.x/arguments/#variadic-arguments).
- When the user did not specify an option, print possible values for it and
  return instead of raising an exception. For example, print the subfile names
  in an H5 file if no subfile name was specified. This allows the user to make
  selections incrementally.
- When the user did not specify an output file, write the output to `sys.stdout`
  if possible instead of raising an exception. This allows the user to use pipes
  and chain commands if they want, or add a quick `-o` option to write to a
  file.
- Always use Python's `logging` module over plain `print` statements. This
  allows the user to control the verbosity.
