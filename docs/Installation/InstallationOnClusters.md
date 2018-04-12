\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Installation on Clusters {#installation_on_clusters}

The installation instructions are the same for most systems because we use shell
scripts to set up the environment for each supercomputer. We describe the
generic installation instructions once, and only note special instructions if
necessary for the particular system. If you have already built SpECTRE and just
want to load the modules, source the shell file for your system and run
`spectre_load_modules`.

## General Instructions

1. Run `export SPECTRE_HOME=/path/to/where/you/want/to/clone`
2. Clone SpECTRE using `git clone SPECTRE_URL $SPECTRE_HOME`
3. Run `cd $SPECTRE_HOME && mkdir build && cd build`
4. Run `. $SPECTRE_HOME/support/Environments/SYSTEM_TO_RUN_ON_gcc.sh`, where
   `SYSTEM_TO_RUN_ON` is replaced by the name of the system as described in the
   relevant section below.
5. If you haven't already, choose where you want to install the dependencies,
   e.g. into `SPECTRE_DEPS` and run `spectre_setup_modules SPECTRE_DEPS`. This
   will take a while to finish. Near the end the command will tell you how to
   load make the modules available by providing a `module use` command.
6. Run `module use SPECTRE_DEPS/modules`
7. Run `spectre_run_cmake`, if you get module loading errors run
   `spectre_unload_modules` and try running `spectre_run_cmake` again. CMake
   should set up successfully.
8. Build the targets you are interested in by running, e.g.
   `make -j4 test-executables`

## BlueWaters at the National Center for Supercomputing Applications

First run `module load bwpy && bwpy-environ`, then follow the general
instructions using `bluewaters` as the `SYSTEM_TO_RUN_ON`.

#### Running tests on BlueWaters

You cannot just run `make test` or `ctest` on BlueWaters.
To run the tests:
1. Get an interactive node using, e.g. `qsub -q debug -I -l nodes=1:ppn=16:xe -l
   walltime=00:30:00` or see the [BlueWaters documentation]
   (https://bluewaters.ncsa.illinois.edu/interactive-jobs)
2. Setup the bwpy environment `module load bwpy && bwpy-environ`
3. Run the `module use` command you did earlier
4. Load the module using the `spectre_load_modules` command
5. Run
   ```
   aprun -n1 -d1 -- bwpy-environ -- \
     $SPECTRE_HOME/build/bin/NonFailureTestsRunTests.sh
   ```

## Cedar, Graham, and Niagara at ComputeCanada

Use `compute_canada` as the `SYSTEM_TO_RUN_ON` in the general instructions.

## Wheeler at Caltech

1. Clone SpECTRE into `$SPECTRE_HOME`
2. Run
   `mkdir $SPECTRE_HOME/build_[gcc|clang] && cd $SPECTRE_HOME/build_[gcc|clang]`
3. Run `. $SPECTRE_HOME/support/Environments/wheeler_[gcc|llvm].env` to load
   the GCC or LLVM/Clang environment
4. Run `cmake -D CMAKE_BUILD_TYPE=[Release|Debug]
   -D CMAKE_Fortran_COMPILER=gfortran $SPECTRE_HOME`
5. Run `make -j4`
