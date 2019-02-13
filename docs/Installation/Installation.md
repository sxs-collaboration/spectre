\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Installation {#installation}

This page details the installation procedure for SpECTRE on personal computers.
For instructions on installing SpECTRE on clusters please refer to the \ref
installation_on_clusters "Installation on clusters" page.

### Dependencies

#### Required:

* [GCC](https://gcc.gnu.org/) 5.4 or later,
[Clang](https://clang.llvm.org/) 3.6 or later, or AppleClang 6.0 or later
* [CMake](https://cmake.org/) 3.3.2 or later
* [Charm++](http://charm.cs.illinois.edu/) 6.8 or newer (must be compiled from source)
* [Git](https://git-scm.com/)
* BLAS (e.g. [OpenBLAS](http://www.openblas.net))
* [Blaze](https://bitbucket.org/blaze-lib/blaze/overview) v3.2
* [Boost](http://www.boost.org/) 1.60.0 or later
* [Brigand](https://github.com/edouarda/brigand)
* [Catch](https://github.com/philsquared/Catch) 2.1.0 or later
* [GSL](https://www.gnu.org/software/gsl/)
* [HDF5](https://support.hdfgroup.org/HDF5/) (non-mpi version on macOS)
* [jemalloc](https://github.com/jemalloc/jemalloc)
* LAPACK
* [libsharp](https://github.com/Libsharp/libsharp) should be built with
  support disabled for openmp and mpi, as we want all of our parallelism to
  be accomplished via Charm++.
* [LIBXSMM](https://github.com/hfp/libxsmm)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) built from a commit more
  recent than November 2016 (versions newer than 0.5.3 should work once any
  are released)
* [Python](https://www.python.org/) 2.7, or 3.5 or later
* [NumPy](http://www.numpy.org/) 1.10 or later
* [SciPy](https://www.scipy.org)

#### Optional:
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) — to generate
  documentation
* [Python](https://www.python.org/) with
  [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/) and
  [Pybtex](https://pybtex.org) — for documentation post-processing
* [Google Benchmark](https://github.com/google/benchmark) - to do
  microbenchmarking inside the SpECTRE framework. v1.2 or newer is required
* [LCOV](http://ltp.sourceforge.net/coverage/lcov.php) and
  [gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html) — to check code test
  coverage
* [coverxygen](https://github.com/psycofdj/coverxygen) — to check documentation
  coverage
* [PAPI](http://icl.utk.edu/papi/) — to access hardware performance counters
* [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) — to format C++
  code in a clear and consistent fashion
* [Clang-Tidy](http://clang.llvm.org/extra/clang-tidy/) — to "lint" C++ code
* [Cppcheck](http://cppcheck.sourceforge.net/) — to analyze C++ code

## Using Docker to obtain a SpECTRE environment

A [Docker](https://www.docker.com/) image is available from
[DockerHub](https://hub.docker.com/r/sxscollaboration/spectrebuildenv/) and can
be used to build SpECTRE on a personal machine.

**Note**: The Docker image is the recommended way of using SpECTRE on a personal
Linux machine. Because of the wide variety of operating systems available today
it is not possible for us to support all configurations. However, using Spack
as outlined below is a supported alternative to Docker images.

**NOTE**: If you have SELinux active
on your system you must figure out how to enable sharing files with the host
OS. If you receive errors that you do not have permission to access a shared
directory it is likely that your system has SELinux enabled. One option is to
disable SELinux at the expense of reducing the security of your system.

To build with the docker image:

1. Clone SpECTRE into SPECTRE_ROOT, a directory of your choice.
2. Retrieve the docker image (you may need `sudo` in front of this command)
   ```
   docker pull sxscollaboration/spectrebuildenv:latest
   ```
3. Start the docker container (you may need `sudo`)
   ```
   docker run -v SPECTRE_ROOT:SPECTRE_ROOT --name CONTAINER_NAME -i -t sxscollaboration/spectrebuildenv:latest /bin/bash
   ```
   (The `--name CONTAINER_NAME` is optional, where CONTAINER_NAME is a name
   of your choice. If you don't name your container, docker will generate an
   arbitrary name.)
   You will end up in a shell in a docker container,
   as root (you need to be root).
   Within the container, SPECTRE_ROOT is available and
   CHARM_DIR is /work/charm. For the following steps, stay inside the docker
   container as root.
4. Make a build directory somewhere inside the container, e.g.
   /work/spectre-build-gcc, and cd into it.
5. Build SpECTRE with
```
   cmake -D CMAKE_Fortran_COMPILER=gfortran-8 \
         -D CHARM_ROOT=/work/charm/multicore-linux64-gcc SPECTRE_ROOT
```
   then
   `make -jN`
   to compile the code, `make test-executables -jN` to compile the test
   executables, and `ctest` to run the tests.

Notes:
  * Everything in your build directory is owned by root, and is
    accessible only within the container.
  * You should edit source files in SPECTRE_ROOT in a separate terminal
    outside the container, and use the container only for compiling and
    running the code.
  * If you exit the container (e.g. ctrl-d),
    your compilation directories are still saved, as is the patch
    that you have applied to /work/charm and any other changes to
    the container that you have made.
    To restart the container, try the following commands
    (you may need `sudo`):
    1. `docker ps -a` # lists all containers and their CONTAINER_IDs and CONTAINER_NAMEs:
    2. `docker start -i CONTAINER_NAME` or `docker start -i CONTAINER_ID` # Restarts your container
  * You can run more than one shell in the same container, for instance
    one shell for compiling with gcc and another for compiling
    with clang.
    To add a new shell, run `docker exec -it CONTAINER_NAME /bin/bash`
    (or `docker exec -it CONTAINER_ID /bin/bash`) from
    a terminal outside the container.
  * In step 3 above, the `-v SPECTRE_ROOT:SPECTRE_ROOT` maps the directory
    SPECTRE_ROOT outside the container to SPECTRE_ROOT inside the container.
    Technically docker allows you to say `-v SPECTRE_ROOT:/my/new/path`
    to map SPECTRE_ROOT outside the container to any path you want inside
    the container, but **do not do this**.  Compiling inside the container
    sets up git hooks in SPECTRE_ROOT that
    contain hardcoded pathnames to SPECTRE_ROOT *as seen from
    inside the container*. So if your source paths inside and outside the
    container are different, commands like `git commit` run *from
    outside the container* will die with `No such file or directory`.
  * To compile with clang, the cmake command is
```
    cmake -D CMAKE_CXX_COMPILER=clang++ \
          -D CMAKE_C_COMPILER=clang \
          -D CMAKE_Fortran_COMPILER=gfortran-8 \
          -D CHARM_ROOT=/work/charm/multicore-linux64-clang SPECTRE_ROOT
```

## Using Singularity to obtain a SpECTRE environment

[Singularity](http://singularity.lbl.gov/index.html) is a container alternative
to Docker with better security and nicer integration.

To use Singularity you must:

1. Build [Singularity](http://singularity.lbl.gov/index.html) and add it to your
   `$PATH`
2. `cd` to the directory where you want to store the SpECTRE Singularity image,
   source, and build directories, let's call it WORKDIR. The WORKDIR must be
   somewhere in your home directory. If this does not work for you, follow the
   Singularity instructions on setting up additional [bind
   points](http://singularity.lbl.gov/docs-mount). Once inside the WORKDIR,
   clone SpECTRE into `WORKDIR/SPECTRE_ROOT`.
3. Run `singularity build spectre.img
   docker://sxscollaboration/spectrebuildenv:latest`.
4. To start the container run `singularity shell spectre.img` and you
   will be dropped into a bash shell. The first thing you will always need to do
   when entering the container is:
   `. SPECTRE_HOME/containers/SingularityLoad.sh`
5. Run `cd SPECTRE_HOME && mkdir build && cd build` to set up a build
   directory.
6. To build SpECTRE run
```
   cmake -D CMAKE_Fortran_COMPILER=gfortran-8 \
         -D CHARM_ROOT=/work/charm/multicore-linux64-gcc SPECTRE_ROOT
```
   followed by
   `make -jN` where `N` is the number of cores to build on in parallel.

Notes:
- You should edit source files in SPECTRE_ROOT in a separate terminal outside
  the container, and use the container only for compiling and running the code.
- Unlike Docker, Singularity does not keep the state between runs. However, it
  shares the home directory with the host OS so you should do all your work
  somewhere in your home directory.
- To run more than one container just do `singularity shell spectre.img` in
  another terminal.
- Since the data you modify lives on the host OS there is no need to worry about
  losing any data, needing to clean up old containers, or sharing data between
  containers and the host.
- To build with clang, the CMake command is:
```
    cmake -D CMAKE_CXX_COMPILER=clang++ \
          -D CMAKE_C_COMPILER=clang \
          -D CMAKE_Fortran_COMPILER=gfortran-8 \
          -D CHARM_ROOT=/work/charm/multicore-linux64-clang SPECTRE_ROOT
```

## Using Spack to set up a SpECTRE environment

SpECTRE's dependencies can be installed with
[Spack](https://github.com/LLNL/spack), a package manager tailored for HPC use.
Install Spack by cloning it into `SPACK_DIR` (a directory of your choice),
then add `SPACK_DIR/bin` to your `PATH`.

For security, it is good practice to make Spack use the system's OpenSSL
rather than allow it to install a new copy — see Spack's documentation for
[instructions](https://spack.readthedocs.io/en/latest/getting_started.html#openssl).
You may need to install the development version of OpenSSL:
* On Ubuntu (16.04), run `sudo apt-get install libssl-dev`
* On Fedora (27), run `sudo dnf install openssl-devel`

Spack works well with a module environment. We recommend
[LMod](https://github.com/TACC/Lmod), which is available on many systems:
* On macOS, install LMod from [brew](https://brew.sh/), then source the LMod
  shell script by adding `. /usr/local/Cellar/lmod/YOUR_VERSION_NUMBER/init/sh`
  to your `.bash_profile`.
* On Ubuntu, run `sudo apt-get install -y lmod` and, for Ubuntu 17.04, add
  `. /etc/profile.d/lmod.sh` to your `.bashrc`. For Ubuntu 16.04, the correct
  path to add is `. /usr/share/lmod/lmod/init/bash`.
* On Arch Linux, run `yaourt -Sy lmod` and add `. /etc/profile.d/lmod.sh` to
  your `.bashrc`,
* On Fedora/RHEL, GNU Environment Modules comes out-of-the-box and works equally
  well.
* Instructions for other Linux distros are available online.

To use modules with Spack, enable Spack's shell support by adding
`. SPACK_DIR/share/spack/setup-env.sh` to your `.bash_profile` or `.bashrc`.

Once you have Spack installed and configured with OpenSSL and LMod, you can
install the SpECTRE dependencies using
```
spack install blaze
spack install brigand@master
spack install libsharp -openmp -mpi
spack install catch
spack install gsl
spack install jemalloc # or from your package manager
spack install libxsmm
spack install yaml-cpp@develop
```
You can also install CMake, OpenBLAS, Boost, and HDF5 from Spack.
To load the packages you've installed from Spack run `spack load PACKAGE`,
or (equivalently) use the `module load` command.

**Note**: Spack allows very flexible configurations and
it is recommended you read the [documentation](https://spack.readthedocs.io) if
you require features such as packages installed with different compilers.

**Note**: On a Mac, you may need to `spack install
yaml-cpp@develop~shared` (note that is a tilde and not a dash in front
of shared) in order to force the building of the static libraries in
order to avoid dynamic linking errors.

### Building SpECTRE

After the dependencies have been installed, Charm++ and SpECTRE can be compiled.
Follow these steps:

1.  Clone [SpECTRE](https://github.com/sxs-collaboration/spectre) into
    `SPECTRE_ROOT`, a directory of your choice.
2.  Install Charm++:
  * Clone [Charm++](http://charm.cs.illinois.edu/software) into `CHARM_DIR`,
    again a directory of your choice.
  * In `CHARM_DIR`, run
    `git checkout v6.8.0` to switch to a supported, stable release of Charm++.
  * Charm++ is compiled by running
    `./build charm++ ARCH OPTIONS`.
    To figure out the correct target architecture and options, you can simply
    run `./build`; the script will then ask you questions to guide you towards
    the correct settings (see notes below for additional details).
    Then compile Charm++.
    The Charm++ build will be located in a new directory,
    `CHARM_DIR/ARCH_OPTS`, whose name may (or may not) have some of the options
    appended to the architecture.
  * The SpECTRE repo contains a patch that must be applied to Charm++ *after*
    Charm++ has been compiled. While still in `CHARM_DIR`, apply this patch by
    running `git apply SPECTRE_ROOT/support/Charm/v6.8.patch`.
  * On macOS 10.12 it is necessary to patch the STL implementation. Insert
    \code
    #ifndef _MACH_PORT_T
    #define _MACH_PORT_T
    #include <sys/_types.h> /* __darwin_mach_port_t */
    typedef __darwin_mach_port_t mach_port_t;
    #include <pthread.h>
    mach_port_t pthread_mach_thread_np(pthread_t);
    #endif /* _MACH_PORT_T */
    \endcode
    into the front of
    `/Library/Developer/CommandLineTools/usr/include/c++/v1/__threading_support`
3.  Return to `SPECTRE_ROOT`, and create a build dir by running
    `mkdir build && cd build`
4.  Build SpECTRE with
    `cmake -D CHARM_ROOT=CHARM_DIR/ARCH_OPTS SPECTRE_ROOT`
    then
    `make -jN`
    to compile the code.
5.  Run the tests with
    `make test-executables && ctest`.

**Notes**:
* For more details on building Charm++, see the directions
  [here](http://charm.cs.illinois.edu/manuals/html/charm++/A.html)
  The correct target is `charm++` and, for a personal machine, the
  correct target architecture is likely to be `multicore-linux64`
  (or `multicore-darwin-x86_64` on macOS).
  On an HPC system, the correct Charm++ target architecture depends on the
  machine's inter-node communication architecture. We will be providing specific
  instructions for various HPC systems.
* Both Charm++ and SpECTRE must be compiled using the same compiler,
  otherwise you will receive undefined reference errors while linking SpECTRE.
  When compiling Charm++ you can specify the compiler using, for example,
  ```
  ./build charm++ ARCH clang
  ```
  When compiling SpECTRE you can specify the compiler to CMake using,
  for example,
  ```
  cmake -D CMAKE_CXX_COMPILER=clang++ \
        -D CMAKE_C_COMPILER=clang \
        -D CMAKE_Fortran_COMPILER=gfortran \
        -D CHARM_ROOT=CHARM_DIR/ARCH_OPTS SPECTRE_ROOT
  ```
* Inside the SpECTRE build directory, use `make list` to see all available
  targets. This list can be refreshed by running CMake again.

## Code Coverage Analysis

For any coverage analysis you will need to have LCOV installed on the system.
For documentation coverage analysis you will also need to install
[coverxygen](https://github.com/psycofdj/coverxygen) and for test coverage
analysis [gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html).

If you have these installed (which is already done if
you are using the docker container), you can look at code coverage as follows:

1. On a gcc build, pass `-D COVERAGE=ON` to `cmake`
2. `make unit-test-coverage`
3. The output is in `docs/html/unit-test-coverage`.
