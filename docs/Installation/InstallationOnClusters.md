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

\note Sample submit scripts for some systems are available in
`support/SubmitScripts`.

## General Instructions

1. Run `export SPECTRE_HOME=/path/to/where/you/want/to/clone`
2. Clone SpECTRE using `git clone SPECTRE_URL $SPECTRE_HOME`
3. Run `cd $SPECTRE_HOME && mkdir build && cd build`
4. Run `. $SPECTRE_HOME/support/Environments/SYSTEM_TO_RUN_ON_gcc.sh`, where
   `SYSTEM_TO_RUN_ON` is replaced by the name of the system as described in the
   relevant section below.
5. If you haven't already installed the dependencies, run
   `export SPECTRE_DEPS=/path/to/where/you/want/the/deps`
   Then run `spectre_setup_modules $SPECTRE_DEPS`. This
   will take a while to finish. Near the end the command will tell you how to
   make the modules available by providing a `module use` command. Make
   sure you are providing an absolute path to `spectre_setup_modules`.
6. Run `module use $SPECTRE_DEPS/modules`
7. Run `spectre_run_cmake`, if you get module loading errors run
   `spectre_unload_modules` and try running `spectre_run_cmake` again. CMake
   should set up successfully.
8. Build the targets you are interested in by running, e.g.
   `make -j4 test-executables`

## BlueWaters at the National Center for Supercomputing Applications

First run `module load bwpy && bwpy-environ`, then follow the general
instructions using `bluewaters` as the `SYSTEM_TO_RUN_ON`. Note that once the
installation is completed, you will still need to run
`module load bwpy && bwpy-environ` in order to run SpECTRE. Also note that
adding this command to your `.bashrc` may cause BlueWaters to hang; you must
run this command from the command line.

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

Follow the general instructions using `wheeler` for `SYSTEM_TO_RUN_ON`, except
you do not need to install any dependencies, so you can skip steps 5 and 6. You
can optionally compile using LLVM/Clang by sourcing `wheeler_clang.sh` instead
of `wheeler_gcc.sh`

## Ocean at Fullerton

Follow the general instructions, using `ocean` for `SYSTEM_TO_RUN_ON`,
you do not need to install any dependencies, so you can skip steps 5 and 6.

## Orca at Fullerton

Follow the general instructions, using `orca` for `SYSTEM_TO_RUN_ON`,
you do not need to install any dependencies, so you can skip steps 5 and 6.

## Zwicky at Fullerton

Follow the general instructions using `zwicky` for `SYSTEM_TO_RUN_ON`, except
you do not need to install any dependencies, so you can skip steps 5 and 6.
Only gcc is supported (use `zwicky_gcc.sh`).

Note that this is a very old machine, so you'll need to load many modules to
do common things (e.g. load modules for openssh and git operations). Do
`module avail` for a list of available modules, and look at
`/home/geoffrey/.bash_profile` for an example list of modules to load.
