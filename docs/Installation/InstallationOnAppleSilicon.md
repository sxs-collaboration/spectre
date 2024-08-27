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

### 1. Clone spectre

Clone the SpECTRE repository in a directory of your choice:

```sh
git clone git@github.com:sxs-collaboration/spectre.git
export SPECTRE_HOME=$PWD/spectre
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

Most of spectre's dependencies can be installed using the
[Homebrew](https://brew.sh) package manager. First, if you haven't already,
install Homebrew by following the instructions on the
[Homebrew](https://brew.sh) homepage. Then, run the following to install a
Fortran compiler and other dependencies:

```
brew install gcc autoconf automake ccache cmake
brew install boost catch2 doxygen gsl hdf5 openblas yaml-cpp
```

\note We use OpenBLAS instead of Apple's Accelerate framework here because
Accelerate fails to reach the same floating point accuracy as OpenBLAS in some
of our tests (specifically partial derivatives).

### 4. Install remaining dependencies

Here, install the remaining dependencies that cannot be installed with Homebrew.
You can install them from source manually, or use the
[Spack](https://github.com/spack/spack) package manager (see below).

#### Install manually

```sh
export SPECTRE_DEPS_ROOT=$HOME/apps
mkdir -p $SPECTRE_DEPS_ROOT
cd $SPECTRE_DEPS_ROOT

# Install Charm++
git clone https://github.com/UIUC-PPL/charm
pushd charm
git checkout v7.0.0
git apply $SPECTRE_HOME/support/charm/v7.0.0.patch
./build charm++ multicore-darwin-arm8 --with-production -g3 -j --build-shared
popd

# The following dependencies are optional! They will be installed in the build
# directory automatically if needed in the next step. You can install them
# manually like this if you want control over where or how they are installed.
# If you don't care, you can skip ahead.

# Install Blaze
mkdir blaze
pushd blaze
curl -L https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz \
> blaze-3.8.tar.gz
tar -xf blaze-3.8.tar.gz
mv blaze-3.8 include
popd

# Install libxsmm
# Need master branch of libxsmm to support Apple Silicon
git clone https://github.com/hfp/libxsmm.git
pushd libxsmm
make
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
# Install dependencies
spack install \
  charmpp@7.0.0: +shared backend=multicore build-target=charm++ \
  blaze@3.8.2 ~blas ~lapack smp=none \
  libxsmm@1.16.1: \
```

### 5. Configure and build SpECTRE

Create a build directory in a location of your choice, e.g.

```sh
cd $SPECTRE_HOME
mkdir build
cd build
```

Next, configure SpECTRE using the following CMake command. If you installed
dependencies with Spack, you can use `spack find -p` to retrieve the root
directories of the packages and replace them in the command below.
You only need to specify `LIBXSMM_ROOT` and `BLAZE_ROOT` if you installed
those packages yourself above. The option `SPECTRE_FETCH_MISSING_DEPS` will
take care of downloading these if you haven't installed them above.

```sh
cmake \
-D CMAKE_C_COMPILER=clang \
-D CMAKE_CXX_COMPILER=clang++ \
-D CMAKE_Fortran_COMPILER=gfortran \
-D CMAKE_BUILD_TYPE=Debug \
-D BUILD_SHARED_LIBS=ON \
-D MEMORY_ALLOCATOR=SYSTEM \
-D CHARM_ROOT=${SPECTRE_DEPS_ROOT}/charm/multicore-darwin-arm8 \
-D SPECTRE_FETCH_MISSING_DEPS=ON \
-D SPECTRE_TEST_TIMEOUT_FACTOR=5 \
-D BLAS_ROOT=$(brew --prefix openblas) \
-D LAPACK_ROOT=$(brew --prefix openblas) \
-D LIBXSMM_ROOT=${SPECTRE_DEPS_ROOT}/libxsmm/ \
-D BLAZE_ROOT=${SPECTRE_DEPS_ROOT}/blaze/ \
..
```

Finally, build and test SpECTRE. E.g., on a Mac with 10 cores,

```sh
make -j10 unit-tests
make -j10 test-executables
ctest --output-on-failure -j10
```

Optionally, to install the python bindings in your python environment,

```sh
make -j10 all-pybindings
pip install -e ${SPECTRE_HOME}/build/bin/python
```
