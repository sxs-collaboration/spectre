\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing Unit Tests {#writing_unit_tests}

Unit tests are placed in the appropriate subdirectory of `tests/Unit`, which
mirrors the directory hierarchy of `src`. The tests are all compiled into
individual libraries to keep link time of testing executables low. Typically
there should be one test library for each production code library. For example,
we have a `DataStructures` library and a `Test_DataStructures` library. When
adding a new test there are several scenarios that can occur, which are outlined
below.

- You are adding a new source file to an existing test library:<br>
  If you are adding a new source file in a directory that already has a
  `CMakeLists.txt` simply create the source file, which should be named
  `Test_ProductionCodeFileBeingTest.cpp` and add that to the `LIBRARY_SOURCES`
  in the `CMakeLists.txt` file in the same directory you are adding the `cpp`
  file.<br>
  If you are adding a new source file to a library but want to place it in a
  subdirectory you must first create the subdirectory. To provide a concrete
  example, say you are adding the directory `TensorEagerMath` to
  `tests/Unit/DataStructures`. After creating the directory you must add a call
  to `add_subdirectory(TensorEagerMath)` to
  `tests/Unit/DataStructures/CMakeLists.txt` *before* the call to
  `add_test_library` and *after* the `LIBRARY_SOURCES` are set. Next add the
  file `tests/Unit/DataStructures/TensorEagerMath/CMakeLists.txt`, which should
  add the new source files by calling `set`, e.g.
  ```
  set(LIBRARY_SOURCES
      ${LIBRARY_SOURCES}
      Test_ProductionCodeFileBeingTest.cpp
      PARENT_SCOPE)
  ```
  The `PARENT_SCOPE` flag tells CMake to make the changes visible in the
  CMakeLists.txt file that called `add_subdirectory`. You can now add the
  `Test_ProductionCodeFileBeingTested.cpp` source file.
- You are adding a new directory:<br>
  If the directory is a new lowest level directory you must add a
  `add_subdirectory` call to `tests/Unit/CMakeLists.txt`. If it is a new
  subdirectory you must add a `add_subdirectory` call to the
  `CMakeLists.txt` file in the directory where you are adding the
  subdirectory. Next you should read the part on adding a new test library.
- You are adding a new test library:<br>
  After creating the subdirectory for the new test library you must add a
  `CMakeLists.txt` file. See `tests/Unit/DataStructures/CMakeLists.txt` for
  an example of one. The `LIBRARY` and `LIBRARY_SOURCES` variables set the name
  of the test library and the source files to be compiled into it. The library
  name should be of the format `Test_ProductionLibraryName`, for example
  `Test_DataStructures`. The library sources should be only the source files in
  the current directory. The `add_subdirectory` command can be used to add
  source files in subdirectories to the same library as is done in
  `tests/Unit/CMakeLists.txt`. The `CMakeLists.txt` in
  `tests/Unit/DataStructures/TensorEagerMath` is an example of how to add source
  files to a library from a subdirectory of the library. Note that the setting
  of `LIBRARY_SOURCES` here first includes the current `LIBRARY_SOURCES` and at
  the end specifies `PARENT_SCOPE`. The `PARENT_SCOPE` flag tells CMake to
  modify the variable in a scope that is visible to the parent directory,
  i.e. the `CMakeLists.txt` that called `add_subdirectory`.<br>
  Finally, in the `CMakeLists.txt` of your new library you must call
  `add_test_library`. Again, see `tests/Unit/DataStructures/CMakeLists.txt` for
  an example. The `add_test_library` function adds a test library with the name
  of the first argument and the source files of the third argument. The second
  argument is the path of the library's directory relative to `tests/Unit`. For
  example, for `Test_DataStructures` it is simply `DataStructures`. The fourth
  and final argument to `add_test_library` are the libraries that must be
  linked. Typically this should only be the production library you're
  testing. For example, `Test_DataStructures` should specify only
  `DataStructures` as the library to link. If you are testing a header-only
  "library" then you do not link any libraries (they must be linked in by the
  libraries actually testing your dependencies). In this case the last argument
  should be `"" # Header-only, link dependencies included from testing lib`

All tests must start with
```cpp
// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"
```
The file `tests/Unit/TestingFramework.hpp` must always be the first include in
the test file and must be separated from the STL includes by a blank line. All
classes and free functions should be in an anonymous/unnamed namespace, e.g.
```cpp
namespace {
class MyFreeClass {
  /* ... */
};

void my_free_function() noexcept {
  /* ... */
}
}  // namespace
```
This is necessary to avoid symbol redefinition errors during linking.

Test cases are added by using the `SPECTRE_TEST_CASE` macro. The first argument
to the macro is the test name, e.g. `"Unit.DataStructures.Tensor"`, and the
second argument is a list of tags. The tags list is a string where each element
is in square brackets. For example, `"[Unit][DataStructures]"`. The tags should
only be the type of test, in this case `Unit`, and the library being tested, in
this case `DataStructures`. The `SPECTRE_TEST_CASE` macro should be treated as a
function, which means that it should be followed by `{ /* test code */ }`. For
example,
\snippet Test_Tensor.cpp example_spectre_test_case
From within a `SPECTRE_TEST_CASE` you are able to do all the things you would
normally do in a C++ function, including calling other functions, setting
variables, using lambdas, etc.

The `CHECK` macro in the above example is provided by
[Catch2](https://github.com/catchorg/Catch2) and is used to check conditions. We
also provide the `CHECK_ITERABLE_APPROX` macro which checks if two `double`s or
two iterable containers of `double`s are approximately
equal. `CHECK_ITERABLE_APPROX` is especially useful for comparing `Tensor`s,
`DataVector`s, and `Tensor<DataVector>`s since it will iterate over nested
containers as well.

\warning Catch's `CHECK` statement only prints numbers out to approximately 10
digits at most, so you should generally prefer `CHECK_ITERABLE_APPROX` for
checking double precision numbers, unless you want to check that two numbers are
bitwise identical.

All unit tests must finish within a few seconds, the hard limit is 5, but having
unit tests that long is strongly discouraged. They should typically complete in
less than half a second. Tests that are longer are often no longer testing a
small enough unit of code and should either be split into several unit tests or
moved to an integration test.

#### Discovering New and Renamed Tests

When you add a new test to a source file or rename an existing test the change
needs to be discovered by the testing infrastructure. This is done by building
the target `rebuild_cache`, e.g. by running `make rebuild_cache`.

#### Testing Pointwise Functions

Pointwise functions should generally be tested in two different ways. The first
is by taking input from an analytic solution and checking that the computed
result is correct. The second is to use the random number generation comparison
with Python infrastructure. In this approach the C++ function being tested is
re-implemented in Python and the results are compared. Please follow these
guidelines:

- The Python implementation should be in a file with the same name as the source
  file that is being re-implemented and placed in the same directory as its
  corresponding `Test_*.cpp` source file.
- The functions should have the same names as the C++ functions they
  re-implement.
- If a function does sums over tensor indices then
  [`numpy.einsum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html)
  should be used in Python to provide an alternative implementation of the loop
  structure.
- You can import Python functions from other re-implementations in the
  `tests/Unit/` directory to reduce code duplication. Note that the path you
  pass to `pypp::SetupLocalPythonEnvironment` determines the directory from
  which you can import Python modules. Either import modules directly from the
  `tests/Unit/` directory (e.g. `import
  PointwiseFunction.GeneralRelativity.Christoffel as christoffel`) or use
  relative imports like `from . import Christoffel as christoffel`. Don't assume
  the Python environment is set up in a subdirectory of `tests/Unit/`.

It is possible to test C++ functions that return by value and ones that return
by `gsl::not_null`. In the latter case, since it is possible to return multiple
values, one Python function taking all non-`gsl::not_null` arguments must be
supplied for each `gsl::not_null` argument to the C++. To perform the test the
`pypp::check_with_random_values()` function must be called. For example, the
following checks various C++ functions by calling into `pypp`:

\snippet Test_Pypp.cpp cxx_two_not_null

The corresponding Python functions are:

\snippet PyppPyTests.py python_two_not_null

#### Testing Failure Cases

Adding the "attribute" `// [[OutputRegex, Regular expression to match]]`
before the `SPECTRE_TEST_CASE` macro will force ctest to only pass the
particular test if the regular expression is found. This can be used to test
error handling. When testing `ASSERT`s you must mark the `SPECTRE_TEST_CASE` as
`[[noreturn]]`,
add the macro `ASSERTION_TEST();` to the beginning of the test, and also have
the test call `ERROR("Failed to trigger ASSERT in an assertion test");` at the
end of the test body.
For example,

```cpp
// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.DataStructures.DataVector.ref_diff_size",
                               "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DataVector data{1.43, 2.83, 3.94, 7.85};
  DataVector data_ref;
  data_ref.set_data_ref(data);
  DataVector data2{1.43, 2.83, 3.94};
  data_ref = data2;
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
```
If the `ifdef SPECTRE_DEBUG` is omitted then compilers will correctly flag
the code as being unreachable which results in warnings.

You can also test `ERROR`s inside your code. These tests need to have the
`OutputRegex`, and also call `ERROR_TEST();` at the beginning. The do not need
the `ifdef SPECTRE_DEBUG` block, they can just call have the code that triggers
an `ERROR`. For example,

\snippet Test_AbortWithErrorMessage.cpp error_test_example

### Building and Running A Single Test File

In cases where low-level header files are frequently being altered and the
changes need to be tested, building `RunTests` becomes extremely time
consuming. The `RunSingleTest` executable in `tests/Unit/RunSingleTest` allows
one to compile only a select few of the test source files and only link in the
necessary libraries. To set which test file and libraries are linked into
`RunSingleTest` edit the `tests/Unit/RunSingleTest/CMakeLists.txt`
file. However, do not commit your changes to that file since it is meant to
serve as an example. To compile `RunSingleTest` use `make RunSingleTest`, and to
run it use `BUILD_DIR/bin/RunSingleTest Unit.Test.Name`.

\warning
`Parallel::abort` does not work correctly in the `RunSingleTest` executable
because a segfault occurs inside Charm++ code after the abort message is
printed.
