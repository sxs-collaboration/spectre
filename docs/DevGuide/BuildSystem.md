\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Build System {#spectre_build_system}

SpECTRE uses [CMake](https://cmake.org/) for the build system. In this section
of the guide we outline how to add new external dependencies, unit tests, and
executables.

## Adding Dependencies

To add a dependency first add a `SetupDEPENDENCY.cmake` file to the `cmake`
directory. You should model this after one of the existing ones for Catch or
Brigand if you're adding a header-only library and yaml-cpp if the library
is not header-only. If CMake does not already support `find_package` for the
library you're adding you can write your own. These should be modeled after
`FindBrigand` or `FindCatch` for header-only libraries, and `FindYAMLCPP`
for compiled libraries. Be sure to test both that setting `LIBRARY_ROOT`
works correctly for your library, and also that if the library is required
that CMake fails if the library is not found.

## Adding Unit Tests

We use the [Catch](https://github.com/philsquared/Catch) testing framework for
unit tests. All unit tests are housed in `tests/Unit` with subdirectories for
each subdirectory of `src`. Add the `cpp` file to the appropriate subdirectory
and also to the `CMakeLists.txt` in that subdirectory. Inside the source file
you can create a new test by adding a
`SPECTRE_TEST_CASE("Unit.Dir.Component", "[Unit][Dir][Tag]")`. The `[Tag]` is optional
and you can have more than one, but the tags should be used quite sparingly.
The purpose of the tags is to be able to run all unit tests or all tests of
a particular set of components, e.g. `ctest -L Data` to run all tests inside
the `Data` directory. Please see other unit tests and the
[Catch documentation](https://github.com/philsquared/Catch) for more help on
writing tests. There will also be a section in the dev guide on how to write
effective tests.  Unit tests should take as short a time as possible, with a
goal of less than two seconds.
 
You can check the unit test coverage of your code by installing all the optional
components and then running `make unit-test-coverage` (after re-running CMake).
This will create the
directory `BUILD_DIR/docs/html/unit-test-coverage/` which is where the coverage
information is located. Open the `index.html` file in your browser and make
sure that your tests are indeed checking all lines of your code. Your pull
requests might not be merged until your line coverage is over 90% (we are aiming
for 100% line coverage wherever possible). Unreachable lines of code can be
excluded from coverage analysis by adding the inline comment `LCOV_EXCL_LINE`
or a block can be excluded using `LCOV_EXCL_START` and `LCOV_EXCL_STOP`.
However, this should be used extremely sparingly since unreachable code paths
should be removed from the code base altogether.

## Adding Executables

All general executables are found in `src/Executables`. To add a new executable
add a directory for it, and that subdirectory to
`src/Executables/CMakeLists.txt` and then model the `CMakeLists.txt` and other
files after one of the existing executables.
