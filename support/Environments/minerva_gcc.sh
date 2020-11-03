#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Instructions to **compile** spectre on Minerva:
#
# 1. We recommend you compile spectre on a compute node to avoid disrupting the
#    login node for other users. This is how you can request a compute node on
#    the `devel` queue for a few hours:
#    ```sh
#    srun -p devel --nodes=1 --ntasks-per-node=16 --time=08:00:00 --pty bash -i
#    ```
#    You will be dropped right into a shell on the compute node as soon as the
#    scheduler has allocated one for you so you can proceed with these
#    instructions.
# 2. Clone the spectre repository, if you haven't already. A good place is in
#    your `/home` directory on Minerva:
#    ```sh
#    cd /home/yourname
#    git clone git@github.com:yourname/spectre.git
#    ```
# 3. Set the `$SPECTRE_HOME` environment variable to the location of the spectre
#    repository, e.g. `/home/yourname/spectre`:
#    ```sh
#    export SPECTRE_HOME=path/to/spectre
#    ```
# 4. Source this script and setup modules:
#    ```sh
#    source $SPECTRE_HOME/support/Environments/minerva_gcc.sh`
#    spectre_setup_modules
#    ```
#    Note: Add steps 3 and 4 to your `.bashrc` file if you don't want to repeat
#    them every time you log in. The `spectre_setup_modules` function only
#    adjusts your `MODULEPATH` to make the installed modules visible but doesn't
#    load any, so it's safe to add to your `.bashrc`.
# 5. Create a build directory, if you don't have one already. A good place is in
#    your `/work` directory on Minerva:
#    ```sh
#    cd /work/yourname
#    mkdir spectre-build
#    ```
#    Note: Add a timestamp or descriptive labels to the name of the build
#    directory, since you may create more build directories later, e.g.
#    `build_YYYY-MM-DD` or `build-clang-Debug`.
# 6. Run `cmake` to configure the build directory:
#    ```sh
#    cd path/to/build/directory
#    module purge
#    spectre_run_cmake
#    ```
#    Note: Remember to `module purge` to work in a clean environment, unless you
#    have reasons not to.
# 7. Compile! With the build directory configured, this script sourced and
#    modules set up, you can skip the previous steps from now on.
#    ```sh
#    module purge
#    spectre_load_modules
#    make -j16 SPECTRE_EXECUTABLE
#    ```
#    Replace `SPECTRE_EXECUTABLE` with the target you want to build, e.g.
#    `unit-tests` or `SolvePoisson3D`.
#
# Instructions to **run** spectre executables on Minerva:
#
# 1. Create a run directory. A good place is in your `/scratch` directory on
#    Minerva. Make sure to choose a descriptive name, e.g.
#    `/scratch/yourname/spectre/your_project/00_the_run`.
# 2. Copy `$SPECTRE_HOME/support/SubmitScripts/Minerva.sh` to the run directory
#    and edit it as the comments in that file instruct.
# 3. Submit the job to Minerva's queue:
#    ```sh
#    sbatch Minerva.sh
#    ```

spectre_setup_modules() {
    export MODULEPATH="\
/home/SPACK2021/share/spack/modules/linux-centos7-haswell:$MODULEPATH"
    export MODULEPATH="\
/home/nfischer/spack/share/spack/modules/linux-centos7-haswell:$MODULEPATH"
}

spectre_load_modules() {
    module load gcc-10.2.0-gcc-10.2.0-vaerku7
    module load binutils-2.36.1-gcc-10.2.0-wtzd7wm
    source /home/nfischer/spack/var/spack/environments/spectre_2021-03-18/loads
    export CHARM_ROOT="\
/home/nfischer/spack/opt/spack/linux-centos7-haswell/gcc-10.2.0/\
charmpp-6.10.2-2waqdh24tz4yt5ozy5clhx6ahxjgivwz"
    # Libsharp is installed separately with `-fPIC` since the Spack package
    # doesn't support that option (yet)
    export LIBSHARP_ROOT="/work/nfischer/spectre/libsharp_2021-03-18/auto"
}

spectre_unload_modules() {
    echo "Unloading a subset of modules is not supported."
    echo "Run 'module purge' to unload all modules."
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake \
      -D CMAKE_C_COMPILER=gcc \
      -D CMAKE_CXX_COMPILER=g++ \
      -D CMAKE_Fortran_COMPILER=gfortran \
      -D CHARM_ROOT=$CHARM_ROOT \
      -D LIBSHARP_ROOT=$LIBSHARP_ROOT \
      -D CMAKE_BUILD_TYPE=Release \
      -D DEBUG_SYMBOLS=OFF \
      -D BUILD_SHARED_LIBS=ON \
      -D MEMORY_ALLOCATOR=SYSTEM \
      -D BUILD_PYTHON_BINDINGS=ON \
      -Wno-dev "$@" $SPECTRE_HOME
}
