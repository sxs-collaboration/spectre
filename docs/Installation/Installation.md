\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Installation {#installation}

\tableofcontents

This page details the installation procedure for SpECTRE on personal computers
using x86-64 processors. For instructions on installing SpECTRE on clusters
please refer to the \subpage installation_on_clusters page. For instructions on
installing SpECTRE on Apple Silicon Macs, please refer to the
\subpage installation_on_apple_silicon page. Refer to the
\subpage versioning_and_releases page for information on specific versions
to install.

### Install a specific version with Spack

You can install SpECTRE with the [Spack](https://github.com/spack/spack) package
manager:

```sh
git clone https://github.com/spack/spack
source ./spack/share/spack/setup-env.sh
spack compiler find
spack external find
spack install spectre executables=ExportCoordinates3D \
  ^charmpp backend=multicore
```

You probably want to customize your installation, e.g., to select a particular
version of SpECTRE, the executables you want to install, additional options such
as Python bindings, or the Charm++ backend. You can display all possible options
with:

```sh
spack info spectre  # or charmpp, etc.
```

Refer to the [Spack documentation](https://spack.readthedocs.io/en/latest/) for
more information.

### Quick-start guide for code development with Docker and Visual Studio Code

If you're new to writing code for SpECTRE and would like to jump right into a
working development environment, a good place to start is our
\subpage dev_guide_quick_start_docker_vscode. If you prefer setting up your
development environment differently, read on!

### Dependencies

**Note**: You don't need to install any of these dependencies by hand,
or by using yum, apt, or other package managers; it is much easier
to instead use Singularity, Docker, or Spack (see the corresponding
sections below) to obtain an environment that includes
all of these dependencies.

#### Required:

* [GCC](https://gcc.gnu.org/) 9.1 or later,
[Clang](https://clang.llvm.org/) 10.0 or later, or AppleClang 11.0.0 or later
* [CMake](https://cmake.org/) 3.18.0 or later
* [Charm++](http://charm.cs.illinois.edu/) 7.0.0, or later (experimental)
  \cite Charmpp1 \cite Charmpp2 \cite Charmpp3
* [Git](https://git-scm.com/)
* BLAS (e.g. [OpenBLAS](http://www.openblas.net))
* [Blaze](https://bitbucket.org/blaze-lib/blaze/overview) v3.8. It can be
  beneficial to install Blaze with CMake so some configuration options are
  determined automatically, such as cache sizes. \cite Blaze1 \cite Blaze2
* [Boost](http://www.boost.org/) 1.60.0 or later
* [Brigand](https://github.com/edouarda/brigand) at commit
  1c398e4f1e817ab195e4cd6fbb03c18cb386eea3 (late 2020) or later
* [Catch](https://github.com/catchorg/Catch2) 2.8.0 or later, but not 3.x as
  SpECTRE doesn't support v3 yet (If installing from source, it is easiest to
  use single-header installation)
* [GSL](https://www.gnu.org/software/gsl/) \cite Gsl
* [HDF5](https://support.hdfgroup.org/HDF5/) (non-mpi version on macOS)
  \cite Hdf5
* [jemalloc](https://github.com/jemalloc/jemalloc)
* LAPACK
* [libsharp](https://github.com/Libsharp/libsharp) should be built with
  support disabled for openmp and mpi, as we want all of our parallelism to
  be accomplished via Charm++. \cite Libsharp
* [LIBXSMM](https://github.com/hfp/libxsmm) version 1.16.1 or later
  \cite Libxsmm
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) version 0.6.3 or later.
  Building with shared library support is also recommended. \cite Yamlcpp
* [Python](https://www.python.org/) 3.7 or later.
* Python dependencies listed in `support/Python/requirements.txt`.
  Install with `pip3 install -r support/Python/requirements.txt`.
  Make sure you are working in a [Python venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
  before installing packages.
  Alternatively, you can set `BOOTSTRAP_PY_DEPS=ON` when configuring a build
  with CMake to install missing Python packages into the build directory
  automatically.
  <details>
  \include support/Python/requirements.txt
  </details>

#### Optional:
* [Pybind11](https://pybind11.readthedocs.io) 2.6.0 or later for SpECTRE Python
  bindings \cite Pybind11
* [Doxygen](https://www.doxygen.nl/index.html) 1.9.1 or later — to
  generate documentation
* Python dev dependencies listed in `support/Python/dev_requirements.txt`
  — for documentation pre- and post-processing, formatting code, etc.
  Install with `pip3 install -r support/Python/dev_requirements.txt`.
  Make sure you are working in a [Python venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
  before installing packages.
  <details>
  \include support/Python/dev_requirements.txt
  </details>
* [Google Benchmark](https://github.com/google/benchmark) - to do
  microbenchmarking inside the SpECTRE framework. v1.2 or newer is required
* [LCOV](http://ltp.sourceforge.net/coverage/lcov.php) and
  [gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html) — to check code test
  coverage
* [PAPI](http://icl.utk.edu/papi/) — to access hardware performance counters
* [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) — to format C++
  code in a clear and consistent fashion
* [Clang-Tidy](http://clang.llvm.org/extra/clang-tidy/) — to "lint" C++ code
* [Cppcheck](http://cppcheck.sourceforge.net/) — to analyze C++ code
* [Scotch](https://gitlab.inria.fr/scotch/scotch) - to build the `ScotchLB`
  graph partition based load balancer in charm++.
* [ffmpeg](https://www.ffmpeg.org/) - for animating 1d simulations with
  matplotlib
* [xsimd](https://github.com/xtensor-stack/xsimd) - for manual vectorization
* [libbacktrace](https://github.com/ianlancetaylor/libbacktrace) - to show
  source files and line numbers in backtraces of errors and asserts. Available
  by default on many systems, so you may not have to install it at all. The
  CMake configuration will tell you if you have libbacktrace installed.
* [ParaView](https://www.paraview.org/) - for visualization \cite Paraview1
  \cite Paraview2
* [SpEC](https://www.black-holes.org/code/SpEC.html) - to load SpEC data.
  Compile the exporter in SpEC's `Support/ApplyObservers/Exporter/` directory
  (see the `Makefile` in that directory). Also make sure to compile SpEC with
  the same compiler and MPI as SpECTRE to avoid compatibility issues.

## Clone the SpECTRE repository

First, clone the [SpECTRE repository](https://github.com/sxs-collaboration/spectre)
to a directory of your choice. In the following we will refer to it as
SPECTRE_ROOT. You may `git clone` from GitHub, in which case SPECTRE_ROOT will
be `<your_current_directory>/spectre`. That is, inside SPECTRE_ROOT are `docs`,
`src`, `support`, `tests` etc. You can also download the source and extract them
to your desired working directory, making sure not to leave out hidden files
when you `cp` or `mv` the source files.

## Using Docker to obtain a SpECTRE environment {#docker_install}

A [Docker](https://www.docker.com/) image is available from
[DockerHub](https://hub.docker.com/r/sxscollaboration/spectre/) and can
be used to build SpECTRE on a personal machine.

**Note**: The Docker image or the Singularity image (see below) are
the recommended ways of using SpECTRE on a personal
Linux machine. Because of the wide variety of operating systems available today
it is not possible for us to support all configurations. However, using Spack
as outlined below is a supported alternative to Docker or Singularity images.

**Note**: If you have SELinux active
on your system you must figure out how to enable sharing files with the host
OS. If you receive errors that you do not have permission to access a shared
directory it is likely that your system has SELinux enabled. One option is to
disable SELinux at the expense of reducing the security of your system.

To build with the Docker image:

1. Install [Docker-Desktop](https://docs.docker.com/get-docker/). For Linux, if
   you want to be able to run the following steps without `sudo`, follow the
   [post-installation-guide](https://docs.docker.com/engine/install/linux-postinstall/)
   to add a non-root user.

2. Retrieve the Docker image (you may need `sudo` in front of this command)
   ```
   docker pull sxscollaboration/spectre:dev
   ```
3. Start the Docker container (you may need `sudo`)
   ```
   docker run -v $SPECTRE_ROOT/:$SPECTRE_ROOT/ --name spectre_dev \
              -i -t sxscollaboration/spectre:dev /bin/bash
   ```
   - `-v $SPECTRE_ROOT/:$SPECTRE_ROOT/` binds the directory `$SPECTRE_ROOT`
   (which is an environment variable you must set up or just use the actual
   path) outside the container to `$SPECTRE_ROOT` inside the container. In this
   way, files in the `$SPECTRE_ROOT` on your host system (outside the container)
   become accessible within the container through the directory SPECTRE_ROOT
   inside the container. If you wonder why the same SPECTRE_ROOT needs to be
   used for both inside and outside the container, which is why `$SPECTRE_ROOT`
   is repeated in the command above with separated by a colon, please see one of
   the notes below regarding `-v` flag.
   - The `--name spectre_dev` is optional. If you don't name your container,
   docker will generate an arbitrary name.
   - On macOS you can significantly increase the performance of file system
   operations by appending the flag `:delegated` to `-v`, e.g.
   `-v $SPECTRE_ROOT/:$SPECTRE_ROOT/:delegated` (see
   https://docs.docker.com/docker-for-mac/osxfs-caching/).
   - The `-i` flag is for interactive mode, which will drop you into the
   container.
   - It can be useful to expose a port to the host so you can run servers such
   as [Jupyter](https://jupyter.org/index.html) for accessing the Python
   bindings (see \ref spectre_using_python) or a Python web server to view the
   documentation. To do so, append the `-p` option, e.g. `-p 8000:8000`.

   You will end up in a bash shell in the docker container,
   as root (you need to be root).
   Within the container, the files in `$SPECTRE_ROOT` are available and Charm++
   is installed in `/work/charm_7_0_0`. For the following steps, stay inside the
   docker container as root.
4. Proceed with [building SpECTRE](#building-spectre).

**Notes:**
  * Everything in your build directory is owned by root, and is
    accessible only within the container.
  * You should edit source files in SPECTRE_ROOT in a separate terminal
    outside the container, and use the container only for compiling and
    running the code.
  * If you exit the container (e.g. ctrl-d),
    your compilation directories are still saved, as are any other changes to
    the container that you have made.
    To restart the container, try the following commands
    (you may need `sudo`):
    1. `docker ps -a`,
      to list all containers with their CONTAINER_IDs and CONTAINER_NAMEs,
    2. `docker start -i CONTAINER_NAME` or `docker start -i CONTAINER_ID`,
      to restart your container (above, the CONTAINER_NAME was spectre_dev).
  * When the Docker container gets updated, you can stop it with
    `docker stop CONTAINER_NAME`, remove it with `docker rm CONTAINER_NAME`
    and then start at step 2 above to run it again.
  * You can run more than one shell in the same container, for instance
    one shell for compiling with gcc and another for compiling
    with clang.
    To add a new shell, run `docker exec -it CONTAINER_NAME /bin/bash`
    (or `docker exec -it CONTAINER_ID /bin/bash`) from
    a terminal outside the container.
  * In step 4 above, technically docker allows you to say
    `-v $SPECTRE_ROOT/:/my/new/path` to map `$SPECTRE_ROOT` outside the
    container to any path you want inside the container, but **do not do this**.
    Compiling inside the container sets up git hooks in SPECTRE_ROOT that
    contain hardcoded pathnames to SPECTRE_ROOT *as seen from inside the
    container*. So if your source paths inside and outside the container are
    different, commands like `git commit` run *from outside the container* will
    die with `No such file or directory`.
  * If you want to use Docker within VSCode, take a look at our
    [quick start guide](../DevGuide/QuickStartDockerVSCode.md) for using Docker
    with VSCode.

## Using Singularity to obtain a SpECTRE environment

[Singularity](https://sylabs.io) is a container alternative
to Docker with better security and nicer integration.

To build SpECTRE with Singularity you must:

1. Build [Singularity](https://sylabs.io) and add it to your
   `$PATH`
2. `cd` to the directory where you want to store the SpECTRE Singularity image,
   source, and build directories, let's call it WORKDIR. The WORKDIR must be
   somewhere in your home directory. If this does not work for you, follow the
   Singularity instructions on setting up additional [bind
   points](https://sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html)
   (version 3.7. For other versions, see the [docs](https://sylabs.io/docs/)).
   Once inside the WORKDIR, clone SpECTRE into `WORKDIR/SPECTRE_ROOT`.
3. Run `sudo singularity build spectre.img
   docker://sxscollaboration/spectre:dev`.
   You can also use spectre:ci instead of spectre:dev if you want more
   compilers installed.

   If you get the error message that `makesquashfs` did not have enough space to
   create the image you need to set a different `SINGULARITY_TMPDIR`. This can
   be done by running: `sudo SINGULARITY_TMPDIR=/path/to/new/tmp singularity
   build spectre.img docker://sxscollaboration/spectre:dev`. Normally
   `SINGULARITY_TMPDIR` is `/tmp`, but building the image will temporarily need
   almost 8GB of space.

   You can control where Singularity stores the downloaded image files from
   DockerHub by specifying the `SINGULARITY_CACHEDIR` environment variable. The
   default is `$HOME/.singularity/`. Note that `$HOME` is `/root` when running
   using `sudo`.
4. To start the container run `singularity shell spectre.img` and you
   will be dropped into a bash shell.
5. Proceed with [building SpECTRE](#building-spectre).

**Notes:**
- You should edit source files in SPECTRE_ROOT in a separate terminal
  outside the container, and use the container only for compiling and running
  the code.
- If you don't have the same Python version in your environment outside the
  container as the version inside the container, this will create problems
  with git hooks. The Singularity container uses python3.8 by default. Thus, it
  is up to the user to ensure that they are using the same Python version inside
  and outside the container. To use a different Python version in the container
  add `-D Python_EXECUTABLE=/path/to/python` to the cmake command where
  `/path/to/python` is usually `/usr/bin/pythonX` and `X` is the version you
  want.
- Unlike Docker, Singularity does not keep the state between runs. However, it
  shares the home directory with the host OS so you should do all your work
  somewhere in your home directory.
- To run more than one container just do `singularity shell spectre.img` in
  another terminal.
- Since the data you modify lives on the host OS there is no need to worry about
  losing any data, needing to clean up old containers, or sharing data between
  containers and the host.

## Using Spack to set up a SpECTRE environment

SpECTRE's dependencies can be installed with
[Spack](https://github.com/spack/spack), a package manager tailored for HPC use.
[Install Spack](https://spack.readthedocs.io/en/latest/getting_started.html) by
cloning it into `SPACK_DIR` (a directory of your choice). Then, enable Spack's
shell support with `source SPACK_DIR/share/spack/setup-env.sh`. Consider adding
this line to your `.bash_profile`, `.bashrc`, or similar. Refer to [Spack's
getting started guide](https://spack.readthedocs.io/en/latest/getting_started.html)
for more information.

Once you have Spack installed, one way to install the SpECTRE dependencies is
with a [Spack environment](https://spack.readthedocs.io/en/latest/environments.html):

\include support/DevEnvironments/spack.yaml

You can also install the Spack packages listed in the environment file above
with a plain `spack install` if you prefer.

**Notes:**
- Spack allows very flexible configurations and we recommended you read the
  [documentation](https://spack.readthedocs.io) if you require features such as
  packages installed with different compilers.
- For security, it is good practice to make Spack [use the system's
  OpenSSL](https://spack.readthedocs.io/en/latest/getting_started.html#openssl)
  rather than allow it to install a new copy.
- To avoid reinstalling lots of system-provided packages with Spack, use the
  `spack external find` feature and the `--reuse` flag to `spack concretize` (or
  `spack install`). You can also install some of the dependencies with your
  system's package manager in advance, e.g., with `apt` or `brew`. If they are
  not picked up by `spack external find` automatically, register them with Spack
  manually. See the [Spack documentation on external
  packages](https://spack.readthedocs.io/en/latest/build_settings.html#external-packages)
  for details.
- Spack works well with a module environment, such as
  [LMod](https://github.com/TACC/Lmod). See the [Spack documentation on
  modules](https://spack.readthedocs.io/en/latest/module_file_support.html) for
  details.
- On macOS:
  - Brigand has an issue with AppleClang 13 when compiling tests (see
    https://github.com/edouarda/brigand/issues/274). Since it is header-only,
    you can simply clone the [Brigand repository](https://github.com/edouarda/brigand)
    instead of installing it with Spack.

## Building Charm++ {#building-charm}

If you are not using a container, haven't installed Charm++ with Spack, or want
to install Charm++ manually for other reasons, follow the installation
instructions in the [Charm++ repository](https://github.com/UIUC-PPL/charm)
and in their [documentation](https://charm.readthedocs.io/en/latest/quickstart.html#installing-charm).
Here are a few notes:

- Once you cloned the [Charm++ repository](https://github.com/UIUC-PPL/charm),
  run `git checkout v7.0.0` to switch to a supported, stable release of
  Charm++.
- Apply the appropriate patch (if there is one) for the version from
  `${SPECTRE_ROOT}/support/Charm`. For example, if you have Charm++ v7.0.0
  then the patch will be `v7.0.0.patch`.
- Choose the `LIBS` target to compile. This is needed so that we can support the
  more sophisticated load balancers in SpECTRE executables.
- On a personal machine the correct target architecture is likely
  `multicore-linux-x86_64`, or `multicore-darwin-x86_64` on macOS. On an HPC
  system the correct Charm++ target architecture depends on the machine's
  inter-node communication architecture. It might take some experimenting to
  figure out which Charm++ configuration provides the best performance.
- Compile Charm++ with support for shared libraries by appending the option
  `--build-shared` to the `./build` command or pass `BUILD_SHARED=ON` to the
  CMake configuration (see the [Charm++ installation
  instructions](https://github.com/UIUC-PPL/charm#building-dynamic-libraries)).
- When compiling Charm++ you can specify the compiler using, for example,
  ```
  ./build LIBS ARCH clang
  ```

## Building SpECTRE {#building-spectre}

Once you have set up your development environment you can compile SpECTRE.
Follow these steps:

1. Create a build directory where you would like to compile SpECTRE. In the
   Docker container you could create, e.g., `/work/spectre-build`. It can be
   useful to add a descriptive label to the name of the build directory since
   you may create more later, e.g., `build-clang-Debug`. Then, `cd` into the
   build directory.
2. Determine the location of your Charm++ installation. In the Docker container
   it is `/work/charm_7_0_0/multicore-linux-x86_64-gcc` for GCC builds and
   `/work/charm_7_0_0/mpi-linux-x86_64-smp-clang` for clang builds. For Spack
   installations you can determine it with
   `spack location --install-dir charmpp`. We refer to the install directory as
   `CHARM_ROOT` below.
3. In your new SpECTRE build directory, configure the build with CMake:
   ```
   cmake -D CHARM_ROOT=$CHARM_ROOT SPECTRE_ROOT
   ```
   Add options to the `cmake` command to to configure the build, select
   compilers, etc. For instance, to build with clang you may run:
   ```
   cmake -D CMAKE_CXX_COMPILER=clang++ \
         -D CMAKE_C_COMPILER=clang \
         -D CMAKE_Fortran_COMPILER=gfortran \
         -D CHARM_ROOT=$CHARM_ROOT \
         SPECTRE_ROOT
   ```
   See \ref common_cmake_flags for documentation on possible configuration
   options.
4. When cmake configuration is done, you are ready to build target executables.
   - You can see the list of available targets by running `make list` (or `ninja
     list` if you are using the Ninja generator) or by using tab completion.
     Compile targets with `make -jN TARGET` (or `ninja -jN TARGET`), where `N`
     is the number of cores to build on in parallel (e.g. `-j4`). Note that the
     Ninja generator allows you to compile individual source files too.
   - Compile the `unit-tests` target and run `ctest -L unit` to run unit tests.
     Compile `test-executables` and run `ctest` to run all tests, including
     executables. To compile `test-executables` you may have to reduce the
     number of cores you build on in parallel to avoid running out of memory.
   - To use the Python bindings (and CLI entry points), compile the
     `all-pybindings` (or `cli`) target (see \ref spectre_using_python).

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
