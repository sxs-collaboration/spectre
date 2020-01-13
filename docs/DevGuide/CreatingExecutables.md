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

\ref tutorial_minimal_parallel_executable tutorial describes how to
add a new parallel executable.

Another simple example of an executable using Charm++ for
parallelization is in `src/Executables/Examples/HelloWorld`.  In this
example, the only additional phase (besides `Initialization` and
`Exit`) is `Execute`, and the phases are executed in order.
`SingletonHelloWorld` defines a single component `HelloWorld`

\snippet SingletonHelloWorld.hpp executable_example_singleton

which specifies via the `chare_type` type alias that it is a singleton
parallel component which means that only one such object will exist
across all processors used by the executable.  Each component must
define the static function `execute_next_phase` which is executed
during the phases (other than `Initialization` and `Exit`) defined in
the metavariables struct.  In `SingletonHelloWorld`, the
`PrintMessage` action is called during the `Execute` phase.

\snippet  SingletonHelloWorld.hpp executable_example_action

The `PrintMessage` action is executed on whatever process the singleton
component is created upon, and prints a message.

Executables can read in an input file (specified by the `--input-file`
argument) that will be parsed when the executable begins.  %Options
specified in the input file can be used to either place items in the
Parallel::ConstGlobalCache (by specifying tags in the
`const_global_cache_tags` type alias of the metavariables, component
and action structs), to construct items in the db::DataBox of
components during initialization (by specifying tags in the
`initialization_tags` type alias of action struct), or be passed to
the `allocate_array` function of an array component (by specifying
tags in the `allocation_tags` type alias of the component).
`SingletonHelloWorld` specifies a single option

\snippet SingletonHelloWorld.hpp executable_example_options

which a string specifying a name that will be placed into the constant global
cache.  The string is fetched when performing the `PrintMessage` action. Items
in the constant global cache are stored once per node that the executable runs
on. An example input file for `SingletonHelloWorld` can be found in
`tests/InputFiles/ExampleExecutables/SingletonHelloWorld.yaml` and shows how to
specify the options (lines beginning with a `#` are comments and can be
ignored).

Furthermore among the included header files

\snippet  SingletonHelloWorld.hpp executable_example_includes

must be the appropriate header for each parallel component type, which in the
`SingletonHelloWorld` example is `AlgorithmSingleton.hpp`.  Note that
these headers are not in the source tree, but are generated automatically when
the code is compiled.

See [the Parallelization documentation](group__ParallelGroup.html#details)
for more details.

### Executable Using Charm++ with Custom main()

While this is technically possible, it has not been tested. We recommend using
the Charm++ supplied main chare mechanism for the time being.

### Executable Not Using Charm++

An example of an executable that does not use Charm++ for parallelization but
still can use all other infrastructure in SpECTRE is in
`src/Executables/Examples/HelloWorldNoCharm`. Adding a non-Charm++ executable to
SpECTRE mostly follows the standard way of adding an executable using CMake. The
only deviation is that the `CMakeLists.txt` file must tell Charm++ not to add a
`main()` by passing the link flags `-nomain-module -nomain`. This is done using
CMake's `set_target_properties`:

```
set_target_properties(
  ${EXECUTABLE}
  PROPERTIES LINK_FLAGS "-nomain-module -nomain"
  )
```

To add the executable as a target you must use the `add_spectre_executable`
function, which is a light weight wrapper around CMake's `add_executable`.
For example,

```
add_spectre_executable(
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
