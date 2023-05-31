\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Build System {#spectre_build_system}

\tableofcontents

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
  PUBLIC
  <list_of_public_libraries>
  PRIVATE
  <list_of_private_libraries>
  INTERFACE
  <list_of_interface_libraries>
  )
```
where each `<list_of_X_libraries>` is an alphabetized list
of libraries of the form
```
  SomeLibrary
  SomeOtherLibrary
  YetAnotherLibrary
```
The libraries listed under `INTERFACE` are those included in at
least one `.hpp` file in `LIB` but never used in any `.cpp` files
in `LIB`.  The libraries listed under `PRIVATE` are used in
at least one `.cpp` file in `LIB` but not in any `.hpp` file
in `LIB`.  The libraries listed under `PUBLIC` are used in at
least one `.hpp` file and at least one `.cpp` file in `LIB`.
Note that a library counts as being used in a `.cpp` file if the
corresponding `.hpp` file includes it. In other words, list a dependency
as `PRIVATE` if it is needed only to compile the library, but not for
including headers. List a dependency as `INTERFACE` if it is not needed
to compile the library, but is needed for including headers. List a
dependency as `PUBLIC` if it is needed for both.

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
- BLAZE_USE_ALWAYS_INLINE
  - Force Blaze inlining (default is `ON`)
  - If disabled or if the platform is unable to 100% guarantee inlining,
    falls back to `BLAZE_USE_STRONG_INLINE` (see below)
  - Forced inlining reduces function call overhead, and so generally reduces
    runtime. However, it does increase compile time and compile memory usage. It
    is also easier to use a debugger when forced inlining is disabled. If you
    are encountering debugger messages like `function inlined`, then forced
    inlining should be disabled.
- BLAZE_USE_STRONG_INLINE
  - Increase the likelihood of Blaze inlining (default is `ON`)
  - Strong inlining reduces function call overhead, and so generally reduces
    runtime. However, it does increase compile time and compile memory usage. It
    is also easier to use a debugger when strong inlining is disabled. If you
    are encountering debugger messages like `function inlined`, then strong
    inlining should be disabled.
- BOOTSTRAP_PY_DEPS and BOOTSTRAP_PY_DEV_DEPS
  - Install missing Python dependencies into the build directory, as listed in
    `support/Python/requirements.txt` and `support/Python/dev_requirements.txt`,
    respectively. This is an alternative to creating a Python environment and
    installing the packages yourself. If you run into problems with packages
    like h5py, numpy or scipy you can/should still install them yourself to make
    sure they use the correct HDF5, BLAS, etc.
    (default is `OFF`)
- BUILD_PYTHON_BINDINGS
  - Build python libraries to call SpECTRE C++ code from python
    (default is `ON`)
- BUILD_SHARED_LIBS
  - Whether shared libraries are built instead of static libraries
    (default is `OFF`)
- BUILD_TESTING
  - Enable building tests. (default is `ON`)
- BUILD_DOCS
  - Enable building documentation. (default is `ON`)
- DOCS_ONLY
  - Build _only_ documentation (default is `OFF`). Requires `BUILD_DOCS=ON`.
- CHARM_ROOT
  - The path to the build directory of `Charm++`
- CHARM_TRACE_PROJECTIONS
  - Enables tracing with Charm++ projections. Specifically, enables the link
    flag `-tracemode projections`. (default is `OFF`)
- CHARM_TRACE_SUMMARY
  - Enables trace summaries with Charm++ projections. Specifically, enables
    the link flag `-tracemode summary`. (default is `OFF`)
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
- CMAKE_INSTALL_PREFIX
  - Location where the `install` target copies executables, libraries, etc. Make
    sure to set this variable before you `install`, or a default location such
    as `/usr/local` is used.
- COVERAGE
  - Enable code coverage with GCOV and LCOV (default `OFF`)
- DEBUG_SYMBOLS
  - Whether or not to use debug symbols (default is `ON`)
  - Disabling debug symbols will reduce compile time and total size of the build
    directory.
- ENABLE_PROFILING
  - Enables various options to make profiling SpECTRE easier
    (default is `OFF`)
- ENABLE_SPECTRE_DEBUG
  - Defines `SPECTRE_DEBUG` macro to enable `ASSERT`s and other debug
    checks so they can be used in Release builds. That is, you get sanity checks
    and compiler optimizations. You cannot disable the checks in Debug builds,
    so this option has no effect in Debug builds.
    (default is `OFF` in release)
- ENABLE_WARNINGS
  - Whether or not warning flags are enabled (default is `ON`)
- KEEP_FRAME_POINTER
  - Whether to keep the frame pointer. Needed for profiling or other cases
    where you need to be able to figure out what the call stack is.
    (default is `OFF`)
- MACHINE
  - Select a machine that we know how to run on, such as a particular
    supercomputer. A file named MACHINE.yaml must exist in support/Machines and
    a submit script template named MACHINE.sh must exist in
    support/SubmitScripts.
- MEMORY_ALLOCATOR
  - Set which memory allocator to use. If there are unexplained segfaults or
    other memory issues, it would be worth setting `MEMORY_ALLOCATOR=SYSTEM` to
    see if that resolves the issue. It could be the case that different
    third-party libraries accidentally end up using different allocators, which
    is undefined behavior and will result in complete chaos.
    (default is `JEMALLOC`)
- PY_DEV_MODE
  - Enable development mode for the Python package, meaning that Python files
    are symlinked rather than copied to the build directory. Allows to edit and
    test Python code much easier, in particular when it uses compiled Python
    bindings, but doesn't replace CMake placeholders in the Python code such as
    the project version. (default is `OFF`)
- SPEC_ROOT
  - Set to a path to a SpEC installation (the SpEC repository root) to link in
    SpEC libraries. In particular, the SpEC::Exporter library is linked in and
    enables loading SpEC data into SpECTRE. See \ref installation for details.
- SPECTRE_TEST_RUNNER
  - Run test executables through a wrapper.  This might be `charmrun`, for
    example.  (default is to not use one)
- SPECTRE_TEST_TIMEOUT_FACTOR (and specific overrides
  SPECTRE_X_TEST_TIMEOUT_FACTOR for X one of UNIT, STANDALONE, INPUT_FILE, or
  PYTHON)
  - Multiply the timeout for the respective set of tests by this factor (default
    is `1`).
  - This is useful to run tests on slower machines.
- SPECTRE_USE_ALWAYS_INLINE
  - Force SpECTRE inlining (default is `ON`)
  - Forced inlining reduces function call overhead, and so generally reduces
    runtime. However, it does increase compile time and compile memory usage. It
    is also easier to use a debugger when forced inlining is disabled. If you
    are encountering debugger messages like `function inlined`, then forced
    inlining should be disabled.
- STRIP_SYMBOLS
  - Whether or not to strip all symbols (default is `OFF`)
  - If enabled strips all extraneous symbols from libraries and executables,
    further reducing the size of them.
- STUB_EXECUTABLE_OBJECT_FILES
  - Replace object files from executables after linking with empty stubs
    (default is `OFF`)
  - This is useful for drastically reducing the build size in CI, but since the
    object files are replaced with empty stubs will generally cause linking
    problems if used during development.
- STUB_LIBRARY_OBJECT_FILES
  - Replace object files from libraries after linking with empty stubs
    (default is `OFF`)
  - This is useful for drastically reducing the build size in CI, but since the
    object files are replaced with empty stubs will generally cause linking
    problems if used during development.
- UBSAN_INTEGER
  - Whether or not to turn on the undefined behavior sanitizer
    [unsigned integer
    overflow](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html) flag
    (`-fsanitize=integer`) (default is `OFF`)
- UBSAN_UNDEFINED
  - Whether or not to turn on the undefined behavior sanitizer
    [undefined
    behavior](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
    compile flags (`-fsanitize=undefined`) (default is `OFF`)
- UNIT_TESTS_IN_TEST_EXECUTABLES
  - Whether to build the `unit-tests` target as part of the `test-executables`
    target. This is used to build only the non-unit tests in the CI build
    that doesn't use the PCH. (default is `ON`)
- USE_CCACHE
  - Use ccache to cache build output so that rebuilding parts of the source tree
    is faster. The cache will use up space on disk, with the default being
    around 2-5GB. If you are performing a one time build to test something
    specific you should consider disabling ccache in order to avoid removing
    cached files that may be useful in other builds.
    (default is `ON`)
- USE_FORMALINE
  - Write the source tree into HDF5 files written to disk in order to increase
    reproducibility of results.
    (default is `ON`)
- USE_GIT_HOOKS
  - Use git hooks to perform some sanity checks so that small goofs are caught
    before they are committed. These checks are particularly useful because they
    also run automatically on \ref github_actions_guide "CI" and must pass
    before pull requests are merged.
    (default is `ON`)
- USE_IWYU
  - Enable [include-what-you-use (IWYU)](https://github.com/include-what-you-use/include-what-you-use)
    tools. (default is `OFF`)
- USE_LD
  - Override the automatically chosen linker. The options are `ld`, `gold`, and
    `lld`.
    (default is `OFF`)
- USE_PCH
  - Whether or not to use pre-compiled headers (default is `ON`)
  - This needs to be turned `OFF` in order to use
    [include-what-you-use
    (IWYU)](https://github.com/include-what-you-use/include-what-you-use)
- USE_SLEEF
  - Whether to use [Sleef](https://github.com/shibatch/sleef) with Blaze to
    vectorize addition math functions like `sin`, `cos`, and `exp`.
    (default is `OFF`)
  - \note Blaze isn't tested super thoroughly across different architectures so
    there's unfortunately no guarantee that Blaze+Sleef will work everywhere.
- USE_XSIMD
  - Whether to use [xsimd](https://github.com/xtensor-stack/xsimd) with Blaze to
    vectorize addition math functions like `sin`, `cos`, and `exp`.
    Defines the macro `SPECTRE_USE_XSIMD`, which can be check to enable manual
    vectorization where necessary.
    (default is `OFF`)

## CMake targets

In addition to individual simulation executables, the following targets are
available to build with `make` or `ninja`:

- unit-tests
  - Build unit tests, which you can run with `ctest -L unit`. Available if
    `BUILD_TESTING` is `ON` (the default).
- test-executables
  - Build all tests, including executables, so you can run all tests with
    `ctest`. Available if `BUILD_TESTING` is `ON` (the default). To compile
    `test-executables` you may have to reduce the number of cores you build on
    in parallel to avoid running out of memory.
- all-pybindings
  - Build Python bindings. See \ref spectre_using_python for details.
- cli
  - Same as all-pybindings
- install
  - Install targets that have been built to the `CMAKE_INSTALL_PREFIX`. Doesn't
    try to build anything else.

## Checking Dependencies

Getting dependencies of libraries correct is quite difficult. SpECTRE offers the
CMake function `check_spectre_libs_dependencies`, defined in
`cmake/SpectreCheckDependencies.cmake`, to check the dependencies for all
libraries in the `libs` target. Individual target dependencies can be checked
using the `check_target_dependencies` CMake function defined in
`cmake/SpectreCheckTargetDependencies.cmake`. Please see those functions in the
source tree for more details on how to use them.

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
inside the `tools/WrapExecutableLinker.sh` script. Function declarations are
provided in `Utilities/Formaline.hpp` and a small function that writes the
source file to disk is defined in `Utilities/Formaline.cpp`. The first
Formaline-related thing done in `WrapExecutableLinker.sh` is to archive
everything in the source directory tracked by git. Once the archive is created
we run `ld -r -b binary -o object.o src.tar.gz` (with unique names for
`object.o` and `src.tar.gz` for each executable that is built to avoid name
collisions) to generate an object file with the source file encoded from
`_binary_src_tar_gz_start` to `_binary_src_tar_gz_end`. Next we write a C++
source file that defines a function `get_archive` to convert the byte stream
into a `std::vector<char>`. We also encode the output of `printenv`, the various
`PATH` environment variables, and the CMake generated `BuildInfo.txt` file
into the source file. Finally, the generated source file is built during the
linking phase and the object file containing the source archive is linked into
the executable.

To further aid in reproducibility, the `printenv` output and
`BuildInfo.txt` contents are written to HDF5 files as part of the
`h5::Header` object. The archive of the source tree is written using the
`h5::SourceArchive` object and can be extracted by running
```
h5dump -d /src.tar.gz -b LE -o src.tar.gz /path/to/hdf5/file.h5
```
