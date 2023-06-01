\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
## Installing SpECTRE on Apple Silicon {#installation_on_apple_silicon}

\tableofcontents

The following instructions show how to install SpECTRE and its dependencies
on Apple Silicon Macs. Apple Silicon is an arm64 architecture, in contrast
to the x86-64 architecture that SpECTRE usually targets. These instructions
will result in an Apple Silicon native build of SpECTRE.

\note Floating-point exception trapping is not currently
supported on Apple Silicon.

### 0. Install the xcode command-line tools.

Install the xcode command-line tools, which include the clang compiler, etc.
Run the following command in the terminal:

```
xcode-select --install
```

### 1. Clone spectre and make a directory to install prerequisites

First, make a directory to hold some prerequisites that spectre depends on.
Name this directory whatever you like, and set `SPECTRE_DEPS_ROOT` to its value.
These instructions, as an example, set this to the `apps` directory in the
user's home folder.
```
cd $HOME
git clone git@github.com:sxs-collaboration/spectre.git
cd spectre
export SPECTRE_HOME=$(pwd)
export SPECTRE_DEPS_ROOT=$HOME/apps
mkdir $SPECTRE_DEPS_ROOT
cd $SPECTRE_DEPS_ROOT
mkdir src
cd src
```

### 2. Install python dependencies

Spectre depends on python and some python packages. There are different ways to
install an arm64-native python stack. The following instructions show how
to do this using the Python 3 interpreter bundled with macOS.

```sh
cd $SPECTRE_HOME

# Create a Python environment
python3 -m venv ./env --upgrade-deps

# Activate the Python environment
. ./env/bin/activate

# Install Python packages
pip install -r support/Python/requirements.txt \
  -r support/Python/dev_requirements.txt

# Optionally install additional packages you might want, like Jupyter
pip install jupyterlab
```

### 3. Install dependencies with Homebrew

Most of spectre's dependencies beyond python can be installed using the
[homebrew](https://brew.sh) package manager. First, if you haven't
already, install Homebrew by
following the instructions on the [homebrew](https://brew.sh) homepage. Then,
run the following to install a fortran compiler and other dependencies:
```
brew install gcc
brew install boost gsl cmake doxygen
brew install ccache autoconf automake jemalloc hdf5 pybind11 yaml-cpp
```

### 4. Install remaining dependencies

Here, install the remaining dependencies that cannot be installed
with homebrew or miniforge. You can install them from source manually, or use
the [Spack](https://github.com/spack/spack) package manager.

#### Install manually

```
export SPECTRE_DEPS_ROOT=$HOME/apps
```

```
cd $SPECTRE_DEPS_ROOT
mkdir blaze
pushd blaze
curl -L https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz \
> blaze-3.8.tar.gz
tar -xf blaze-3.8.tar.gz
mv blaze-3.8 include
popd

git clone https://github.com/edouarda/brigand.git

# Need master branch of libxsmm to support Apple Silicon
git clone https://github.com/hfp/libxsmm.git
pushd libxsmm
make
popd

pushd ./src
git clone https://github.com/Libsharp/libsharp.git
cd libsharp

# Do not use compiler flag -march=native (unsupported on Apple Silicon)
sed "s/-march=native//" configure.ac > configure.ac.mod
mv configure.ac.mod configure.ac

autoupdate
autoconf
./configure
make
mv auto $SPECTRE_DEPS_ROOT/libsharp
popd
```

Next, clone, patch, and install charm++ v7.0.0.
```
git clone https://github.com/UIUC-PPL/charm
pushd charm
git checkout v7.0.0
./build LIBS multicore-darwin-arm8 --with-production -g3 -j
popd
```
Lastly, install Catch2. Note that catch2 v2.13.7 is required, and homebrew
can only install v3. Catch2 is a header only package, so no need to build
separately.

```
git clone https://github.com/catchorg/Catch2.git
pushd Catch2
git checkout v2.13.7
popd
```

#### Install with Spack

```sh
# Download spack
cd ~
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
cd spack
# Switch to latest release
git checkout releases/latest
# Load shell support
. ./share/spack/setup-env.sh
# Find some system packages so we don't have to install them all from source
spack external find
spack external find python
# Install dependencies
spack install \
  blaze@3.8.2 \
  brigand@master \
  catch2@2.8:2 \
  charmpp@7.0.0: backend=multicore \
  libxsmm@main \
  libsharp~mpi~openmp
```

### 5. Configure and build SpECTRE

Create a build directory in a location of your choice, e.g.
```
cd ${SPECTRE_HOME}
mkdir build
cd build
```

Next, configure SpECTRE using the following CMake command. If you installed
dependencies with Spack, you can use `spack find -p` to retrieve the root
directories of the packages and replace them in the command below.

```
cmake \
-D CMAKE_C_COMPILER=clang \
-D CMAKE_CXX_COMPILER=clang++ \
-D CMAKE_Fortran_COMPILER=gfortran \
-D CMAKE_BUILD_TYPE=Debug \
-D BUILD_SHARED_LIBS=OFF \
-D MEMORY_ALLOCATOR=SYSTEM \
-D CHARM_ROOT=${SPECTRE_DEPS_ROOT}/charm/multicore-darwin-arm8 \
-D SPECTRE_TEST_TIMEOUT_FACTOR=5 \
-D LIBXSMM_ROOT=${SPECTRE_DEPS_ROOT}/libxsmm/ \
-D BLAZE_ROOT=${SPECTRE_DEPS_ROOT}/blaze/ \
-D BRIGAND_ROOT=${SPECTRE_DEPS_ROOT}/brigand/ \
-D LIBSHARP_ROOT=${SPECTRE_DEPS_ROOT}/libsharp/ \
-D CATCH_INCLUDE_DIR=${SPECTRE_DEPS_ROOT}/Catch2/single_include/catch2/ \
..
```

Finally, build and test SpECTRE. E.g., on a Mac with 10 cores,
```
make -j10 unit-tests
make -j10 test-executables
ctest --output-on-failure -j10
```
