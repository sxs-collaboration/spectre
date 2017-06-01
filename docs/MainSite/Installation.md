\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Installation {#installation}

### Requirements:

* [GCC](https://gcc.gnu.org/) 5.2 or later,
[Clang](https://clang.llvm.org/) 3.6 or later, or AppleClang 6.0 or later
* [CMake](https://cmake.org/) 3.3.2 or later
* [Charm++](http://charm.cs.illinois.edu/) (must be compiled from source)
* [Git](https://git-scm.com/)
* BLAS
* [Boost](http://www.boost.org/)
* [Brigand](https://github.com/edouarda/brigand)
* [Catch](https://github.com/philsquared/Catch) (v1.6.1 or older)
* [HDF5](https://support.hdfgroup.org/HDF5/) (non-mpi version on macOS)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)


### Optional:
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) for
documentation generation
* [LCOV](http://ltp.sourceforge.net/coverage/lcov.php) and
[gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html) for checking code coverage
* [coverxygen](https://github.com/psycofdj/coverxygen) for checking
documentation coverage
* [PAPI](http://icl.utk.edu/papi/)
* [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html)
* [Clang-Tidy](http://clang.llvm.org/extra/clang-tidy/)
* [Cppcheck](http://cppcheck.sourceforge.net/)

### Using Docker to Compile SpECTRE

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

### Installing Dependencies Using Spack

All the dependencies of SpECTRE can be installed from Spack. To setup Spack,
first install LMod on your system. On macOS LMod can be installed via
[brew](https://brew.sh/) and then sourcing the LMod shell script in your
`~/.bash_profile` by adding
`. /usr/local/Cellar/lmod/YOUR_VERSION_NUMBER/init/sh`. On Ubuntu LMod can
be installed by running `sudo apt-get install -y lmod` and then adding
`. /etc/profile.d/lmod.sh` to your `~/.bashrc` file. On Arch Linux run
`yaourt -Sy lmod` and add `. /etc/profile.d/lmod.sh` to your `~/.bashrc`.
Instructions for other Linux distros are available online.

Clone [Spack](https://github.com/LLNL/spack) and add `spack/bin` to your path.
Run `openssl version` and note the version number, for example `1.1.0e`.
Next run `which openssl` and note the path, for example `/usr/bin`.
Open `~/.spack/package.yaml` (you may need to create it) in your favourite
text editor and add
```
packages:
    openssl:
        paths:
            openssl@1.1.0e: /usr
        buildable: False
```
substituting your version number and path in.

Once you have Spack installed and configured with OpenSSL and LMod you can
install the dependencies using
```
spack install brigand@master
spack install catch
spack install yaml-cpp@develop
```

Next, run `ls spack/share/spack/modules` add one entry of
```
module use /path/to/spack/share/spack/modules/YOUR_OS_VERSION_INFO
```
for each `YOUR_OS_VERSION_INFO` inside `spack/share/spack/modules` to your
`~/.bash_profile` or `~.bashrc`.

You can also install CMake, OpenBLAS, Boost, and HDF5 from Spack.
To load the packages you've installed from Spack run `spack load PACKAGE`,
or use the `module load` command. Spack allows very flexible configurations and
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
