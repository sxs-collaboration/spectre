\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Installation {#installation}

### Dependencies

#### Required:

* [GCC](https://gcc.gnu.org/) 5.2 or later,
[Clang](https://clang.llvm.org/) 3.6 or later, or AppleClang 6.0 or later
* [CMake](https://cmake.org/) 3.3.2 or later
* [Charm++](http://charm.cs.illinois.edu/) (must be compiled from source)
* [Git](https://git-scm.com/)
* BLAS (e.g. [OpenBLAS](http://www.openblas.net))
* [Blaze](https://bitbucket.org/blaze-lib/blaze/overview)
* [Boost](http://www.boost.org/)
* [Brigand](https://github.com/edouarda/brigand)
* [Catch](https://github.com/philsquared/Catch) v1.6.1 or older
* [GSL](https://www.gnu.org/software/gsl/)
* [HDF5](https://support.hdfgroup.org/HDF5/) (non-mpi version on macOS)
* [jemalloc](https://github.com/jemalloc/jemalloc)
* [LIBXSMM](https://github.com/hfp/libxsmm)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)

#### Optional:
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) — to generate
  documentation
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

#### Using Docker to Compile SpECTRE

A [Docker](https://www.docker.com/) image is available from
[DockerHub](https://hub.docker.com/r/sxscollaboration/spectrebuildenv/) and can
be used to build SpECTRE on a personal machine. To retrieve the Docker image run

```
docker pull sxscollaboration/spectrebuildenv:latest
```

The image already has Charm++ configured, all that needs to be done is to
clone SpECTRE and compile it.

**Note**: The Docker image is the recommended way of using SpECTRE on a personal
Linux machine. Because of the wide variety of operating systems available today
it is not possible for us to support all configurations. However, using Spack
as outlined below is a supported alternative to Docker images.

#### Installing Dependencies Using Spack

SpECTRE's dependencies can be installed with
[Spack](https://github.com/LLNL/spack), a package manager tailored for HPC use.
Install Spack by cloning it into `SPACK_DIR` (a directory of your choice),
then add `SPACK_DIR/bin` to your `PATH`.

For security, it is good practice to make Spack use the system's OpenSLL
rather than allow it to install a new copy — see Spack's documentation for
[instructions](https://spack.readthedocs.io/en/latest/getting_started.html#openssl).

Spack works well with a module environment. We recommend
[LMod](https://github.com/TACC/Lmod), which is available on many systems:
* On macOS, install LMod from [brew](https://brew.sh/), then source the LMod
  shell script by adding `. /usr/local/Cellar/lmod/YOUR_VERSION_NUMBER/init/sh`
  to your `.bash_profile`.
* On Ubuntu, run `sudo apt-get install -y lmod` and add
  `. /etc/profile.d/lmod.sh` to your `.bashrc`.
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
spack install catch@1.6.1
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

### Building SpECTRE

* Ensure you have all dependencies installed. See
"Installing Dependencies Using Spack" for one method.
* Install [Charm++](http://charm.cs.illinois.edu/software)
 for your machine. Further details
 [here](http://charm.cs.illinois.edu/manuals/html/charm++/A.html)
* Clone [SpECTRE](https://github.com/sxs-collaboration/spectre)
* Apply the Charm++ patch for your version *after* building Charm++ by running
`git apply SPECTRE_ROOT/support/Charm/vx.y.z.patch` in the Charm++ directory
* `mkdir build && cd build`
* The compiler used for Charm++ and SpECTRE must be the same, otherwise you will
  receive undefined references errors during linking.
  You will only need to worry about this if you explicitly specified a
  compiler when building Charm++. When compiling SpECTRE you
  can specify the compiler to CMake using, for example
  ```
  cmake -D CMAKE_CXX_COMPILER=clang++ \
        -D CMAKE_C_COMPILER=clang \
        -D CMAKE_Fortran_COMPILER=gfortran \
        -D CHARM_ROOT=/path/to/charm/BUILD_DIR SPECTRE_ROOT
  ```
* `cmake -D CHARM_ROOT=/path/to/charm/BUILD_DIR SPECTRE_ROOT`
* `make list` to see all available targets
* Run tests by running `make %RunTests && ctest`

### Code Coverage Analysis

For any coverage analysis you will need to have LCOV installed on the system.
For documentation coverage analysis you will also need to install
[coverxygen](https://github.com/psycofdj/coverxygen) and for test coverage
analysis [gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html).
