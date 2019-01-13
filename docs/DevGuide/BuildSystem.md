\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Build System {#spectre_build_system}

# CMake {#cmake}

SpECTRE uses [CMake](https://cmake.org/) for the build system. In this
section of the guide we outline how to [add new source
files](#adding_source_files), [libraries](#adding_libraries), [unit
tests](#adding_unit_tests), [executables](#adding_executables), and
[external dependencies](#adding_external_dependencies).  We also
describe [commonly used CMake flags](#common_cmake_flags).

Note that in editing `CMakeLists.txt` files, it is conventional to
indent multiline commands by two spaces (except for the first line),
and to separate most commands by blank lines.

## Adding Source Files {#adding_source_files}

SpECTRE organizes source files into subdirectories of `src` that are
compiled into libraries.  To add a new source file `FILE.cpp` to an
existing library in `src/PATH/DIR`, just edit
`src/PATH/DIR/CMakeLists.txt` and add `FILE.cpp` to the list of files
in
```
set(LIBRARY_SOURCES
  <list_of_files>
  )
```
such that the resulting `<list_of_files>` is in alphabetical order.

### Adding Libraries {#adding_libraries}

To add a source file `FILE.cpp` that is compiled into a new library `LIB` in a
directory `src/PATH/DIR` (either in a new directory, or in an existing
directory that either does not have a `CMakeLists.txt` file, or does
not create a library in the existing `CMakeLists.txt`):
- Create (if necessary) a `CMakeLists.txt` file in `DIR`, with the following
two lines at the top:
```
# Distributed under the MIT License.
# See LICENSE.txt for details.
```
- In the parent directory (i.e. `src/PATH`), (if necessary) add the
following line to its `CMakeLists.txt` file (if necessary, recursively
do the previous step and this one until you reach a `CMakeLists.txt` that
adds the appropriate subdirectory):
```
add_subdirectory(DIR)
```
If there are already other `add_subdirectory()` lines in the file, place
the new one so that the subdirectories are in alphabetical order.
- Add the line:
```
set(LIBRARY LIB)
```
where convention is that `LIB` = `DIR`.  As library names must be
unique, this is not always possible, in which case the convention is to
prepend the parent directory to `DIR`.
- Add the lines
```
set(LIBRARY_SOURCES
  FILE.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  <list_of_interfaces>
  )
```
where `<list_of_interfaces>` is an alphabetized list of lines of the form
```
  INTERFACE SOME_LIBRARY
```
where `SOME_LIBRARY` is a library that must be linked in order for the
new library to be successfully linked in an executable.

## Adding Unit Tests {#adding_unit_tests}

We use the [Catch](https://github.com/philsquared/Catch) testing
framework for unit tests. All unit tests are housed in `tests/Unit`
with subdirectories for each subdirectory of `src`. Add the `cpp` file
to the appropriate subdirectory and also to the `CMakeLists.txt` in
that subdirectory. Inside the source file you can create a new test by
adding a `SPECTRE_TEST_CASE("Unit.Dir.Component",
"[Unit][Dir][Tag]")`. The `[Tag]` is optional and you can have more
than one, but the tags should be used quite sparingly.  The purpose of
the tags is to be able to run all unit tests or all tests of a
particular set of components, e.g. `ctest -L Data` to run all tests
inside the `Data` directory. Please see \ref writing_unit_tests
"writing unit tests", other unit tests and the [Catch
documentation](https://github.com/philsquared/Catch) for more help on
writing tests. Unit tests should take as short a time as possible,
with a goal of less than two seconds.  Please also limit the number of
distinct cases (by using `SECTION`s).

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

## Adding Executables {#adding_executables}

All general executables are found in `src/Executables`, while those
for specific evolution (elliptic) systems are found in
`src/Evolution/Executables` (`src/Elliptic/Executables`).  See \ref
dev_guide_creating_executables "how to create executables".

## Adding External Dependencies {#adding_external_dependencies}

To add an external dependency, first add a `SetupDEPENDENCY.cmake`
file to the `cmake` directory. You should model this after one of the
existing ones for `Catch` or `Brigand` if you're adding a header-only
library and `yaml-cpp` if the library is not header-only. If CMake
does not already support `find_package` for the library you're adding
you can write your own. These should be modeled after `FindBrigand` or
`FindCatch` for header-only libraries, and `FindYAMLCPP` for compiled
libraries. The `SetupDEPENDENCY.cmake` file must then be included in
the root `spectre/CMakeLists.txt`. Be sure to test both that setting
`LIBRARY_ROOT` works correctly for your library, and also that if the
library is required that CMake fails gracefully if the library is not
found.

## Commonly Used CMake flags {#common_cmake_flags}
The following are the most common flags used to control building with
`CMake`. They are used by
```
cmake -D FLAG1=OPT1 ... -D FLAGN=OPTN <SPECTRE_ROOT>
```
- ASAN
  - Whether or not to turn on the address sanitizer compile flags
    (`-fsanitize=address`) (default is `OFF`)
- BUILD_SHARED_LIBS
  - Whether shared libraries are built instead of static libraries
    (default is `OFF`)
- CHARM_ROOT
  - The path to the build directory of `Charm++`
- CMAKE_BUILD_TYPE
  - Sets the build type.  Common options:
    - `Debug` (the default if the flag is not specified): sets flags
      that trigger additional error checking
    - `Release`
- CMAKE_C_COMPILER
  - The `C` compiler used (defaults to whatever is determined by
    `CMake/Modules/CMakeDetermineCCompiler.cmake`, usually `cc`)
- CMAKE_CXX_COMPILER
  - The `C++` compiler used (defaults to whatever is determined by
    `CMake/Modules/CMakeDetermineCXXCompiler.cmake`, usually `c++`)
- CMAKE_Fortran_COMPILER
  - The `Fortran` compiler used (defaults to whatever is determined by
    `CMake/Modules/CMakeDetermineFortranCompiler.cmake`)
- CMAKE_C_FLAGS
  - Additional flags passed to the `C` compiler.
- CMAKE_CXX_FLAGS
  - Additional flags passed to the `C++` compiler.
- CMAKE_Fortran_FLAGS
  - Additional flags passed to the `Fortran` compiler.
- CMAKE_RUNTIME_OUTPUT_DIRECTORY
  - Sets the directory where the library and executables are placed.
    By default libraries end up in `<BUILD_DIR>/lib` and executables
    in `<BUILD_DIR>/bin`.
- ENABLE_WARNINGS
  - Whether or not warning flags are enabled (default is `ON`)
- UBSAN_INTEGER
  - Whether or not to turn on the undefined behavior sanitizer
    unsigned integer overflow flag (`-fsanitize=integer`) (default is
    `OFF`)
- UBSAN_UNDEFINED
  - Whether or not to turn on the undefined behavior sanitizer
    undefined behavior compile flags (`-fsanitize=undefined`) (default
    is `OFF`)
- USE_PCH
  - Whether or not to use pre-compiled headers (default is `ON`)
  - This needs to be turned `OFF` in order to use
    [include-what-you-use
    (IWYU)](https://github.com/include-what-you-use/include-what-you-use)

## Formaline

SpECTRE's implementation of Formaline is based on, but distinct in
implementation from, the original design by
[Erik Schnetter and Christian Ott](https://github.com/hypercott/formaline),
which embeds an archive of the source tree into the executable. The original
design creates a C/C++ file with a function that returns an array/vector of
`char`s (a byte stream). However, this results in a very large source file (50MB
or more), which is very slow to compile and ends up more than doubling the link
time. Instead, SpECTRE's Formaline implementation uses the linker `ld` to
encode a file into an object, which means
rather than creating a large source file, we can directly encode the source tree
archive into the binary at the linking stage.

Most of SpECTRE's Formaline is implemented
inside the `tools/WrapLinker.sh` script. Function declarations are provided
in `Utilities/Formaline.hpp` and a small function that writes the source
file to disk is defined in `Utilities/Formaline.cpp`. The first
Formaline-related thing done in `WrapLinker.sh` is to archive everything in the
source directory tracked by git. Once the archive is created we run
`ld -r -b binary -o object.o src.tar.gz` (with unique names for `object.o` and
`src.tar.gz` for each executable that is built to avoid name collisions) to
generate an object file with the source file encoded from
`_binary_src_tar_gz_start` to `_binary_src_tar_gz_end`. Next we write a C++
source file that defines a function `get_archive` to convert the byte stream
into a `std::vector<char>`. We also encode the output of `printenv`, the various
`PATH` environment variables, and the CMake generated `LibraryVersions.txt` file
into the source file. Finally, the generated source file is built during the
linking phase and the object file containing the source archive is linked into
the executable.

To further aid in reproducibility, the `printenv` output and
`LibraryVersions.txt` contents are written to HDF5 files as part of the
`h5::Header` object. The archive of the source tree is written using the
`h5::SourceArchive` object and can be extracted by running
```
h5dump -d /src.tar.gz -b LE -o src.tar.gz /path/to/hdf5/file.h5
```
