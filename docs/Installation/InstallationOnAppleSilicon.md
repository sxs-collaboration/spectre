\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
## Installing SpECTRE on Apple Silicon {#installation_on_apple_silicon}

The following instructions show how to install SpECTRE and its dependencies
on Apple Silicon Macs. Apple Silicon is an arm64 architecture, in contrast
to the x86-64 architecture that SpECTRE usually targets. These instructions
will result in an Apple Silicon native build of SpECTRE.

\note Floating-point exception trapping is not currently
supported on Apple Silicon.

### 0. Install the xcode command-line tools.

Install the xcode command-line tools, which include the clang compiler, etc.

```
xcode-select --install
```

### 1. Make a directory to install prerequisites

First, make a directory to hold some prerequisites that spectre depends on.
Name this directory whatever you like, and set `SPECTRE_DEPS_ROOT` to its value.
These instructions, as an example, set this to the `apps` directory in the
user's home folder.
```
export SPECTRE_DEPS_ROOT=$HOME/apps
mkdir $SPECTRE_DEPS_ROOT
cd $SPECTRE_DEPS_ROOT
mkdir src
cd src
```

### 2. Install python dependencies

Spectre depends on python and some python packages. There are different ways to
install an arm64-native python stack. The following instructions show how
to do this using [Miniforge](https://github.com/conda-forge/miniforge),
which supports Apple Silicon natively.

#### Miniforge
Distribution Miniforge is a way to install an arm64-native python stack on Apple
Silicon Macs. Download the
[Miniforge3-MacOSX-arm64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)
installation script, and then do the following:
```
# Install miniforge. Accept all default choices
bash Miniforge3-MacOSX-arm64.sh
# Activate conda base environment for current session
eval "$(${HOME}/miniforge3/bin/conda shell.zsh hook)"
# Activate conda at startup (to make sure you're always using native python)
conda init zsh
exit # close shell. conda will automatically load when new zsh shells start.
```

Open a new terminal window. Then, install the necessary python packages.
```
conda install numpy scipy matplotlib h5py pyyaml
```

You might also wish to install jupyter notebook support, using
```
conda install -c conda-forge jupyterlab
```

### 3. Install dependencies with Homebrew

Most of spectre's dependencies beyond python can be installed using the
[homebrew](https://brew.sh) package manager. First, if you haven't
already, install Homebrew by
following the instructions on the [homebrew](https://brew.sh) homepage. Then,
run the following to install a fortran compiler and other dependencies:
```
brew install gcc
brew install openblas boost gsl cmake doxygen
brew install ccache autoconf automake jemalloc hdf5 pybind11 yaml-cpp
```

### 4. Install remaining dependencies

Here, install the remaining dependencies that cannot be installed
with homebrew or miniforge.

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
cd Libsharp

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

Next, install charm++. Note that Apple Silicon Macs require v7.0.0.

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

### 5. Clone, configure, and build SpECTRE.
Next, clone SpECTRE, make a build directory, and configure. In whatever
directory you prefer, clone SpECTRE and make a build directory, e.g.
```
git clone git@github.com:sxs-collaboration/spectre.git
cd spectre
mkdir build
cd build
```

Next, configure SpECTRE using the following `cmake command`.

```
cmake -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -D \
CMAKE_Fortran_COMPILER=gfortran -D BUILD_PYTHON_BINDINGS=ON \
-D MEMORY_ALLOCATOR=SYSTEM \
-D CHARM_ROOT=${SPECTRE_DEPS_ROOT}/charm/multicore-darwin-arm8 \
-D BLAS_ROOT=$(brew --prefix openblas) -D CMAKE_BUILD_TYPE=Debug \
-D DEBUG_SYMBOLS=ON -D USE_PCH=ON \
-D SPECTRE_TEST_TIMEOUT_FACTOR=5 \
-D LIBXSMM_ROOT=${SPECTRE_DEPS_ROOT}/libxsmm/ \
-D BLAZE_ROOT=${SPECTRE_DEPS_ROOT}/blaze/ \
-D BRIGAND_ROOT=${SPECTRE_DEPS_ROOT}/brigand/ \
-D LIBSHARP_ROOT=${SPECTRE_DEPS_ROOT}/libsharp/ \
-D CATCH_INCLUDE_DIR=${SPECTRE_DEPS_ROOT}/Catch2/include/ \
-D Boost_ROOT=$(brew --prefix boost)/ \
-D CLANG_TIDY_BIN=$(brew --prefix llvm)/bin/clang-tidy \
-D BUILD_SHARED_LIBS=OFF \
-DGSL_ROOT=$(brew --prefix gsl)/include \
-DGSL_LIBRARY=$(brew --prefix gsl)/lib/libgsl.a ..
```

Finally, build and test SpECTRE. E.g., on a Mac with 10 cores,
```
make -j10
make -j10 test-executables
ctest --output-on-failure -j10
```
