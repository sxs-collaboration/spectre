\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# The minimal SpECTRE executable {#tutorial_minimal_parallel_executable}

This tutorial will illustrate what is needed to compile the minimal
SpECTRE executable that will simply print some useful information
about the executable and then exit.

Specifically, this tutorial will introduce:
- `add_spectre_parallel_executable`, a CMake function that will add a
  build target for a SpECTRE executable
- A user-provided `Metavariables`, a C++ struct that is used to
  specify the metaprogram that is converted into a C++ executable.
- how to build a SpECTRE executable
- how to run a SpECTRE executable
- useful information that can be extracted from all SpECTRE executables
- `Main`, the main parallel component that acts as the main function
  of a C++ executable.

In this tutorial, `SPECTRE_ROOT` refers to the directory in which
SpECTRE was cloned, and `SPECTRE_BUILD_DIR` refers to the directory in
which you built SpECTRE.

### Creating a build target for a parallel executable

The first step is to select (or create) a directory (within
`SPECTRE_ROOT/src`) that will hold the files from which the executable
will be created.  For this tutorial we created the directory
`ParallelTutorial` in `src/Executables/Examples`.  If you create a new
directory for the executable, you will need to edit the
`CMakeLists.txt` file in its parent directory and add (replacing
`ParallelTutorial` with the appropriate directory name)

\snippet Examples/CMakeLists.txt add_subdirectory

The second step is to edit (or create) the `CMakeLists.txt` file in
the executable directory and add a call to
`add_spectre_parallel_executable` such as:

\snippet ParallelTutorial/CMakeLists.txt add_spectre_executable

The (SpECTRE defined) `CMake` function `add_spectre_executable` takes
five arguments:

- The name of the executable (in the given example `MinimalExample`)
  that will be created, which will also be the name of the
  corresponding build target.
- The base name (in the given example `MinimalExecutable`) of the two
  user-provided files that declare (`MinimalExecutableFwd.hpp`) and
  define (`MinimalExecutable.hpp`) a C++ struct (or struct template)
  that will describe the metaprogram that is converted into a C++
  executable.  We will refer to this struct (template) as the
  metavariables struct (template), and the files declaring and
  defining the metavariables struct (template) as the metavariables
  files.
- The path relative to `SPECTRE_ROOT/src` to the directory holding the
  metavariables files (in the given example
  `Executables/Examples/ParallelTutorial`)
- The specific metavariables struct (in the given example
  `Metavariables`) that will be used to create the executable.  This
  is either the name of the metavariables struct in the metavariables
  files, or a specific instantiation of the metavariables struct
  template in the metavariables files.
- A list of SpECTRE libraries that the executable will be linked
  against (in the given example the `CMake` list `LIBS_TO_LINK` where
  `%Informer` and `Utilities` are two SpECTRE libraries)

### Writing the metavariables files

The header files from which an executable is generated must declare
and define a metavariables struct that can be thought of as a
compile-time input file that defines what the executable will do.

The first metavariables file (`MinimalExecutableFwd.hpp`) is simply a
forward declaration of the metavariables struct:

\snippet MinimalExecutableFwd.hpp metavariables_forward_declaration

(Note that `#%pragma once` tells the compiler to only include the file
once per compilation, and the `/// \%cond` and `/// \%endcond` around
the code tells doxygen not to generate documentation from the wrapped
region)

The second metavariables file (`MinimalExecutable.hpp`) holds the
definition of the metavariables struct.

\snippet MinimalExecutable.hpp metavariables_definition

The metavariables struct must define an enum class `Phase` with the
phases of the executable (which must include `Initialization` and
`Exit`), and a static function `determine_next_phase`.  In this
example, no additional phases are defined, and the executable will
execute the `Initialization` phase followed by the `Exit` phase.
(`Parallel::CProxy_GlobalCache` is an unused proxy to the
`GlobalCache` that is explained below)

The metavariables struct must define a type alias `component_list`
that is a `tmpl::list` (a typelist defined in `Utilities/TMPL.hpp`) of
the parallel components used by the executable.  In this example no
parallel components are used.

The metavariables struct must define `help`, a `static constexpr
Options::String` that will be printed as part of the help message of the
executable. (`Options::String` is defined in `Options/Options.hpp`.)

In addition to defining the metavaribles struct, the metavariables
file must define the two vectors of functions `charm_init_node_funcs`
and `charm_init_proc_funcs`

\snippet  MinimalExecutable.hpp executable_example_charm_init

that are executed at startup by Charm++ on each node and processing
element (PE) the executable runs on.  In this example, the vectors are
empty.

### Building a SpECTRE executable

Let `$EXECUTABLE` be the name of the executable (the first argument
passed to the `add_spectre_executable` `CMake` function, so in this
example `MinimalExample`).  Then the executable can be built by
running the following command in `$SPECTRE_BUILD_DIR`:

```
make $EXECUTABLE
```

which will produce the executable of the same name in
`$SPECTRE_BUILD_DIR/bin`.

### Running a SpECTRE executable

To run a SpECTRE executable, run the command:

```
./$SPECTRE_BUILD_DIR/bin/$EXECUTABLE  <options>
```

where `<options>` must include any required command-line options.  In
the simple example for this tutorial, no command-line options are
required.

On a laptop, we get the following output:

```
Charm++: standalone mode (not using charmrun)
Charm++> Running in Multicore mode:  1 threads
Converse/Charm++ Commit ID: v6.8.0-0-ga36028edb
CharmLB> Load balancer assumes all CPUs are same.
Charm++> Running on 1 unique compute nodes (4-way SMP).
Charm++> cpu topology info is gathered in 0.000 seconds.

Executing 'some_path/MinimalExample' using 1 processors.
Date and time at startup: Fri Oct 25 15:03:05 2019

SpECTRE Build Information:
Version:                      0.0.0
Compiled on host:             kosh-3.local
Compiled in directory:        some_path/build
Source directory is:          some_path/parallel_tutorial
Compiled on git branch:       feature/parallel_tutorial
Compiled with git hash:       1f4ab20cbda9ae8072a17df69de9731c43687468
Linked on:                    Fri Oct 25 13:26:06 2019


Done!
CkWallTimer in seconds 0.004185
Date and time at completion: Fri Oct 25 15:03:05 2019

[Partition 0][Node 0] End of program
```

which includes information that will be printed by every SpECTRE
executable on startup and exit.  First, there is information provided
by `Charm++` that will depend upon how `Charm++` was built.  Next you
will see information provided by SpECTRE which includes:

- the name of the executable
- how many processes the executable was run on
- the date and time at startup
- the version of SpECTRE that was used to compile the executable
- where the executable was compiled
- which git branch and hash was used to compile the executable
- when the executable was linked

On exit, the executable will print that it is `Done!`, followed by how
the long the executable took to run as timed by the `Charm++`
wall-clock timer, and the date and time at completion, followed by any
information the `Charm++` provides upon exiting the program.

### Extracting useful information from a SpECTRE executable

Every SpECTRE executable comes with a set of command-line options that
can be used to obtain useful information about the executable (and for
executables expecting an input file, whether or not the input file can
be parsed successfully).

#### Getting a list of available options

To get a list of available options for a SpECTRE executable, run
either of the following commands:

```
./$SPECTRE_BUILD_DIR/bin/$EXECUTABLE  --help
./$SPECTRE_BUILD_DIR/bin/$EXECUTABLE  -h
```

In the middle of the `Charm++` startup information will now appear a
long list of `Charm++` related command-line options which we will
ignore for this tutorial.  After the SpECTRE startup information,
there is now a list of available command-line options:

```
  -h [ --help ]             Describe program options
  --check-options           Check input file options
  --dump-source-tree-as arg If specified, then a gzip archive of the source
                            tree is dumped with the specified name. The archive
                            can be expanded using 'tar -xzf ARCHIVE.tar.gz'
  --dump-paths              Dump the PATH, CPATH, LD_LIBRARY_PATH,
                            LIBRARY_PATH, and CMAKE_PREFIX_PATH at compile
                            time.
  --dump-environment        Dump the result of printenv at compile time.
  --dump-library-versions   Dump the contents of SpECTRE's LibraryVersions.txt
  --dump-only               Exit after dumping requested information.
```

which we will describe in detail below. This is followed by a
description of the expected input-file options, which in this example
will simply print the `help` string from the metavariables struct, and
the statement that there are no options expected:

```
==== Description of expected options:
A minimal executable

<No options>
```

We will cover input file options in a future tutorial.

#### Checking options

As the minimal executable in this tutorial expects no input-file
options, running the command:

```
./$SPECTRE_BUILD_DIR/bin/$EXECUTABLE  --check-options
```

will print out `No options to check!` and exit the program.  See a
future tutorial for an example of checking input-file options.

#### Dumping the source tree

In order to aid in reproducibility, all SpECTRE executables contain a
copy of `$SPECTRE_ROOT`.  To obtain the source tree as an archive
file, run the command:

```
./$SPECTRE_BUILD_DIR/bin/$EXECUTABLE  --dump-source-tree-as SpECTRE
```

which will produce the archive file `SpECTRE.tar.gz` which can be
expanded with the command:

```
tar -xzf SpECTRE.tar.gz
```

#### Dumping other information about a SpECTRE executable

In addition to dumping the entire source tree there are the following options:

- `--dump-paths` will print out various paths from when the
  executable was compiled
- `--dump-environment` will print the results of `printenv`
  (i.e. all of the environment variables) from when the executable was
  compiled
- `--dump-library-versions` will print the contents of SpECTRE's
  `LibraryVersions.txt` which will contain the versions of libraries
  that SpECTRE linked against, as well as various `CMake` variables

Also note that the option `--dump-only` can be used to have the
SpECTRE executable terminate immediately after dumping the
information.

### Behind the scenes:  the main parallel component

Every Charm++ executable needs a mainchare.  All SpECTRE executables
use `Main` (found in `src/Parallel/Main.hpp`) as the mainchare.  When
a SpECTRE executable is run, the constructor of `Main` that is executed
takes the command line options as an argument (as type
`CkArgMsg*`).  This constructor performs the following operations:

- Prints useful startup information
- Parses the command line options, performing any requested operations
- Parses the input file options, populating a tagged tuple with their
  values.  (This is discussed in more detail in a future tutorial on
  input file options.)
- Creates the `GlobalCache` (a nodegroup chare) that holds
  objects created from input file options that are stored
  once per Charm++ node, as well as proxies to all other parallel
  components.
- Creates user-requested non-array parallel components, passing them
  the proxy to the `GlobalCache` as well as any items they
  request that can be created from input file options.
- Creates empty array user-requested parallel components
- Sends the complete list of parallel components to the
  `GlobalCache`.

Once the list of parallel components is sent to the `GlobalCache`
on each node, the `Main` member function
`allocate_array_components_and_execute_initial_phase` will be
executed.  This will allocate the elements of the array parallel
components by calling the `allocate_array` member function of each
component.  Then the `start_phase` member function is called on each
component, which will execute the phase action list for the
`Initialization` phase for each component.  Charm++ will execute each
phase until it detects that nothing is happening (quiescence
detection).  As this represents a global synchronization point, the
number of phases should be minimized in order to exploit the power of
SpECTRE.  After each phase, the `execute_next_phase` member function
of `Main` will be called.  This member function first determines what
the next phase is.  If the next phase is `Exit`, then some useful
information is printed and the program exits gracefully.  Otherwise the
`execute_next_phase` member function of each parallel component is
called.

