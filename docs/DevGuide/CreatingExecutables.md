\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Creating Executables {#dev_guide_creating_executables}

There are several different types of executables that can be built:
- An executable that uses Charm++ to run in parallel,
  e.g. `Executables/ParallelInfo`
- An executable that does not use Charm++, does not run in parallel, and
  supplies its own `main`, e.g. `Executables/Benchmark`
- An executable that uses Charm++ to run in parallel but supplies its own `main`
- An executable that uses custom compilation or linking flags,
  e.g. `DebugPreprocessor`
- Executables used for evolutions or elliptic solves

### Executable Using Charm++ for Parallelization

Pull request #751 will add documentation here.

### Executable Using Charm++ with Custom main()

While this is technically possible, it has not been tested. We recommend using
the Charm++ supplied main chare mechanism for the time being.

### Executable Not Using Charm++

An example of an executable that does not use Charm++ for parallelization but
still can use all other infrastructure in SpECTRE is in
`src/Executables/HelloWorldNoCharm`. Adding a non-Charm++ executable to SpECTRE
mostly follows the standard way of adding an executable using CMake. The only
deviation is that the `CMakeLists.txt` file must tell Charm++ not to add a
`main()` by passing the link flags `-nomain-module -nomain`. This is done using
CMake's `set_target_properties`:

```
set_target_properties(
  ${EXECUTABLE}
  PROPERTIES LINK_FLAGS "-nomain-module -nomain"
  )
```

To add the executable as a target you must use CMake's `add_executable` function
directly, not any of the SpECTRE-provided wrappers that add Charm++ parallelized
executables. For example,

```
add_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL # Exclude from calls to `make` without a specified target
  HelloWorld.cpp
  )
```

You can link in any of the SpECTRE libraries by adding them to the
`target_link_libraries`, for example:

```
target_link_libraries(
  ${EXECUTABLE}
  DataStructures
  )
```

We recommend that you add a test that the executable properly runs by adding an
input file to `tests/InputFiles` in an appropriate subdirectory. See
[`tests/InputFiles/ExampleExecutables/HelloWorldNoCharm.yaml`]
(https://github.com/sxs-collaboration/spectre/tree/develop/tests/InputFiles/
ExampleExecutables/HelloWorldNoCharm.yaml)
for an example.
The input file is passed to the executable using `--input-file
path/to/Input.yaml`. In the case of the executable not taking any input file
this is just used to generate a test that runs the executable.

For these types of executables `main` can take the usual `(int argc, char
*argv[])` and parse command line options. Executables not using Charm++ are just
standard executables that can link in any of the libraries in SpECTRE.

\warning
Currently calling `Parallel::abort` results in a segfault deep inside Charm++
code. However, the error messages from `ASSERT` and `ERROR` are still printed.

### Executable With Custom Compilation or Linking Flags

Use the CMake function `set_target_properties` to add flags to an executable. To
call a completely custom compiler invocation you should use the
`add_custom_target` CMake function. The need for the `custom_target` level of
control is rare and should generally be avoided since it adds quite a bit of
technical debt to the code base. Thus, it is not explained here. If you are
certain you need it you can see the `DebugPreprocessor` executable's
`CMakeLists.txt` file for an example.

### Executable Used for Evolution or Elliptic Solve

Once they are written, see the tutorials specific to evolution and elliptic
solves.
