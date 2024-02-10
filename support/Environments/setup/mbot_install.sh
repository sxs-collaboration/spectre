#!/bin/env bash

# Distributed under the MIT License.
# See LICENSE.txt for details.

set -e

if [ $# != 1 ]; then
    echo "You must pass one argument to spectre_setup_modules, which"
    echo "is the directory where you want the dependencies to be built."
    exit 1
fi

PARALLEL_MAKE_ARG=64
dep_dir=`realpath $1`
mkdir -p $dep_dir
cd $dep_dir

# log all output from this script
exec > >(tee "log.$(date +%F-%T)") 2>&1

mkdir -p $dep_dir/modules

if [ "${SPECTRE_HOME}" = "" ]; then
    echo "You must set SPECTRE_HOME to the SpECTRE source directory"
    exit 1
fi

################################################################
# Autotools Begin
################################################################
# We install autotools globally independent of compiler versions.
# This should be fine since we don't link against autotools, we
# just use it to configure and build 3rd party libs.
cd $dep_dir
_AUTOTOOLS_VERSION=2.72
_AUTOMAKE_VERSION=1.16
_LIBTOOL_VERSION=2.4
LOCATION=$dep_dir/autotools/${_AUTOTOOLS_VERSION}
# We need to set up the module file first since each part of autotools
# needs to be able to use the previous parts.
MODULE_FILE=$dep_dir/modules/global/autotools/${_AUTOTOOLS_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## We use the following versions of autotools:
##  autoconf: ${_AUTOTOOLS_VERSION}
##  automake: ${_AUTOMAKE_VERSION}
##  libtool: ${_LIBTOOL_VERSION}
##
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr "\tPrepends \$apps_path/share/man to MANPATH"
        puts stderr "\tenvironment variables."
        puts stderr ""
        puts stderr "\tThis allows you to use the autotools"
        puts stderr ""
}

module-whatis   "Sets up your environment to use the autotools"

# for Tcl script use only
set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        autotools

prepend-path    PATH            \$apps_path/bin
prepend-path    MANPATH         \$apps_path/share/man
prepend-path  CMAKE_PREFIX_PATH  \$apps_path
EOF
fi

module use $dep_dir/modules/global
module load autotools/2.72
if [ -d ${LOCATION} ]; then
    echo "autotools version ${_AUTOTOOLS_VERSION} already installed"
else
    # We don't install m4 since the system one is reasonably new.
    wget https://ftp.gnu.org/gnu/autoconf/autoconf-${_AUTOTOOLS_VERSION}.tar.gz
    tar xf autoconf-${_AUTOTOOLS_VERSION}.tar.gz
    cd ./autoconf-${_AUTOTOOLS_VERSION}
    ./configure --prefix=${LOCATION}
    make all install
    cd ../
    rm -rf ./autoconf-${_AUTOTOOLS_VERSION} \
       ./autoconf-${_AUTOTOOLS_VERSION}.tar.gz

    wget https://ftp.gnu.org/gnu/automake/automake-${_AUTOMAKE_VERSION}.tar.gz
    tar xf automake-${_AUTOMAKE_VERSION}.tar.gz
    cd ./automake-${_AUTOMAKE_VERSION}
    ./configure --prefix=${LOCATION}
    make all install
    cd ../
    rm -rf ./automake-${_AUTOMAKE_VERSION} \
       ./automake-${_AUTOMAKE_VERSION}.tar.gz

    wget https://ftp.gnu.org/gnu/libtool/libtool-${_LIBTOOL_VERSION}.tar.gz
    tar xf libtool-${_LIBTOOL_VERSION}.tar.gz
    cd ./libtool-${_LIBTOOL_VERSION}
    ./configure --prefix=${LOCATION}
    make all install
    cd ../
    rm -rf ./libtool-${_LIBTOOL_VERSION} ./libtool-${_LIBTOOL_VERSION}.tar.gz
    chmod -R 555 ${LOCATION}
fi
################################################################
# Autotools End
################################################################

################################################################
# pinentry begin
################################################################
_PINENTRY_VERSION=1.2.1
LOCATION=$dep_dir/pinentry/${_PINENTRY_VERSION}
if [ -d ${LOCATION} ]; then
    echo "pinentry version ${_PINENTRY_VERSION} already installed"
else
    wget https://www.gnupg.org/ftp/gcrypt/libgpg-error/libgpg-error-1.47.tar.bz2
    tar xf libgpg-error-1.47.tar.bz2
    cd ./libgpg-error-1.47
    ./configure --prefix=${LOCATION} \
                --enable-static --disable-shared
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./libgpg-error-1.47.tar.bz2 ./libgpg-error-1.47

    wget https://www.gnupg.org/ftp/gcrypt/libassuan/libassuan-2.5.6.tar.bz2
    tar xf libassuan-2.5.6.tar.bz2
    cd libassuan-2.5.6
    ./configure --prefix=${LOCATION} --with-libgpg-error-prefix=${LOCATION} \
                --enable-static --disable-shared
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./libassuan-2.5.6.tar.bz2 ./libassuan-2.5.6

    wget https://www.gnupg.org/ftp/gcrypt/pinentry/pinentry-${_PINENTRY_VERSION}.tar.bz2
    tar xf pinentry-${_PINENTRY_VERSION}.tar.bz2
    cd pinentry-${_PINENTRY_VERSION}
    ./configure --prefix=${LOCATION} \
                --enable-pinentry-tty \
                --enable-inside-emacs=yes \
                --with-libgpg-error-prefix=${LOCATION} \
                --with-libassuan-prefix=${LOCATION} \
                --disable-pinentry-curses \
                --disable-fallback-curses \
                --enable-static --disable-shared
    make -j${PARALLEL_MAKE_ARG}
    rm ${LOCATION}/bin/*
    make install
    cd ../
    rm -rf ./pinentry-${_PINENTRY_VERSION}.tar.bz2 \
       ./pinentry-${_PINENTRY_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=$dep_dir/modules/global/pinentry/${_PINENTRY_VERSION}.lua
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
help([[
  Pinentry v${_PINENTRY_VERSION}
]])
whatis("Sets up your environment so you can use Pinentry (v${_PINENTRY_VERSION})")

local apps_path = "${LOCATION}"

conflict("pinentry")

prepend_path("PATH", pathJoin(apps_path, "/bin"))
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# pinentry end
################################################################

################################################################
# tmux begin
################################################################
_TMUX_VERSION=3.3a
LOCATION=$dep_dir/tmux/${_TMUX_VERSION}
if [ -d ${LOCATION} ]; then
    echo "tmux version ${_TMUX_VERSION} already installed"
else
    wget https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz
    tar xf libevent-2.1.12-stable.tar.gz
    cd ./libevent-2.1.12-stable/
    ./configure --prefix=${LOCATION} --enable-static --disable-shared --with-pic
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./libevent-2.1.12-stable/ \
       ./libevent-2.1.12-stable.tar.gz

    wget https://invisible-island.net/datafiles/release/ncurses.tar.gz
    tar xf ncurses.tar.gz
    cd ./ncurses-*
    ./configure --with-termlib --without-shared --with-pic \
                --enable-pc-files \
                --with-pkg-config-libdir=${LOCATION}/lib/pkgconfig \
                --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./ncurses*

    wget https://github.com/tmux/tmux/archive/refs/tags/${_TMUX_VERSION}.tar.gz
    tar xf ${_TMUX_VERSION}.tar.gz
    cd ./tmux-${_TMUX_VERSION}
    module unload autotools/${_AUTOTOOLS_VERSION}
    ./autogen.sh
    PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${LOCATION}/lib/pkgconfig ./configure \
                   --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    # Remove any binaries from libevent or ncurses since we don't need them.
    # This avoids polluting the PATH.
    rm -f ${LOCATION}/bin/*
    make install
    cd ../
    rm -rf ./${_TMUX_VERSION}.tar.gz ./tmux-${_TMUX_VERSION}
    module load autotools/${_AUTOTOOLS_VERSION}

    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=$dep_dir/modules/global/tmux/${_TMUX_VERSION}.lua
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
help([[
  Tmux v${_TMUX_VERSION}
]])
whatis("Sets up your environment so you can use Tmux (v${_TMUX_VERSION})")

local apps_path = "${LOCATION}"

conflict("tmux")

prepend_path("PATH", pathJoin(apps_path, "/bin"))
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# tmux end
################################################################

################################################################
# Emacs begin
################################################################
_EMACS_VERSION=29.2
LOCATION=$dep_dir/emacs/${_EMACS_VERSION}
if [ -d ${LOCATION} ]; then
    echo "tmux version ${_TMUX_VERSION} already installed"
else
    _BEFORE_CPATH=$CPATH
    _BEFORE_LIBRARY_PATH=$LIBRARY_PATH
    _BEFORE_PKG_CONFIG_PATH=$PKG_CONFIG_PATH
    export CPATH=${LOCATION}/include
    export LIBRARY_PATH=${LOCATION}/lib/:${LOCATION}/lib64
    export PKG_CONFIG_PATH=${LOCATION}/lib/pkgconfig:${LOCATION}/lib64/pkgconfig

    wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
    tar xf gmp-6.3.0.tar.xz
    cd ./gmp-6.3.0
    ./configure --disable-shared --enable-static --with-pic --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./gmp-6.3.0.tar.xz ./gmp-6.3.0

    wget https://github.com/akheron/jansson/releases/download/v2.14/jansson-2.14.tar.gz
    tar xf jansson-2.14.tar.gz
    cd ./jansson-2.14
    ./configure --disable-shared --enable-static --with-pic --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./jansson-2.14.tar.gz ./jansson-2.14

    wget https://sqlite.org/2024/sqlite-autoconf-3450100.tar.gz
    tar xf sqlite-autoconf-3450100.tar.gz
    cd ./sqlite-autoconf-3450100
    # Build sqlite shared, but have Emacs link against the system install.
    # The system install doesn't have the headers, just the library.
    # ./configure --disable-shared --enable-static --with-pic --prefix=${LOCATION}
    ./configure --with-pic --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./sqlite-autoconf*

    wget https://invisible-island.net/datafiles/release/ncurses.tar.gz
    tar xf ncurses.tar.gz
    cd ./ncurses-*
    mkdir -p ${LOCATION}/lib/pkgconfig
    ./configure --with-termlib --without-shared --with-pic --prefix=${LOCATION} \
                --enable-pc-files --with-pkg-config-libdir=${LOCATION}/lib/pkgconfig
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./ncurses*

    # The below are needed (and not fully working) if GNUTLS is desired.
    # wget https://ftp.gnu.org/gnu/nettle/nettle-3.9.1.tar.gz
    # tar xf nettle-3.9.1.tar.gz
    # cd ./nettle-3.9.1
    # ./configure --disable-shared --enable-static --with-pic --prefix=${LOCATION}
    # make -j${PARALLEL_MAKE_ARG}
    # make install
    # cd ../
    # rm -rf ./nettle*

    # wget https://ftp.gnu.org/gnu/libtasn1/libtasn1-4.19.0.tar.gz
    # tar xf libtasn1-4.19.0.tar.gz
    # cd ./libtasn1-4.19.0
    # ./configure --disable-shared --enable-static --disable-valgrind-tests \
        #             --with-pic --prefix=${LOCATION}
    # make -j${PARALLEL_MAKE_ARG}
    # make install
    # cd ../
    # rm -rf ./libtasn1*

    # wget https://github.com/libexpat/libexpat/releases/download/R_2_5_0/expat-2.5.0.tar.bz2
    # tar xf expat-2.5.0.tar.bz2
    # cd ./expat-2.5.0
    # ./configure --disable-shared --enable-static --with-pic --prefix=${LOCATION}
    # make -j${PARALLEL_MAKE_ARG}
    # make install
    # cd ../
    # rm -rf ./expat*

    # wget https://www.nlnetlabs.nl/downloads/unbound/unbound-1.19.0.tar.gz
    # tar xf unbound-1.19.0.tar.gz
    # cd ./unbound-1.19.0
    # ./configure --disable-shared --enable-static --with-pic \
        #             --with-libexpat=${LOCATION} --prefix=${LOCATION}
    # make -j${PARALLEL_MAKE_ARG}
    # make install
    # cd ../
    # rm -rf ./unbound*

    # wget https://github.com/libffi/libffi/archive/refs/tags/v3.4.4.tar.gz
    # tar xf v3.4.4.tar.gz
    # cd libffi-3.4.4
    # ./autogen.sh
    # ./configure --prefix=${LOCATION} --disable-docs --with-pic
    # make -j${PARALLEL_MAKE_ARG}
    # make install
    # cd $dep_dir
    # rm -rf libffi-3.4.4 v3.4.4.tar.gz

    # wget https://github.com/p11-glue/p11-kit/releases/download/0.25.3/p11-kit-0.25.3.tar.xz
    # ./configure --without-bash-completion --prefix=${LOCATION}

    # wget https://www.gnupg.org/ftp/gcrypt/gnutls/v3.7/gnutls-3.7.10.tar.xz

    # Install Emacs!
    wget https://us.mirrors.cicku.me/gnu/emacs/emacs-${_EMACS_VERSION}.tar.xz
    tar xf emacs-${_EMACS_VERSION}.tar.xz
    cd ./emacs-${_EMACS_VERSION}
    ./configure --without-xpm --without-jpeg --without-tiff --without-gif \
                --without-png --without-rsvg --without-webp --without-cairo \
                --without-imagemagick --without-native-image-api \
                --without-selinux --without-gpm --without-gnutls \
                --with-json --with-sqlite3=yes \
                --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    # clear bin directory so we only add emacs to the PATH.
    rm ${LOCATION}/bin/*
    make install
    cd ../
    rm -rf ./emacs-${_EMACS_VERSION}*

    chmod -R 555 ${LOCATION}

    export CPATH=$_BEFORE_CPATH
    export LIBRARY_PATH=$_BEFORE_LIBRARY_PATH
    export PKG_CONFIG_PATH=$_BEFORE_PKG_CONFIG_PATH
fi

MODULE_FILE=$dep_dir/modules/global/emacs/${_EMACS_VERSION}.lua
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
help([[
  Emacs v${_EMACS_VERSION}
]])
whatis("Sets up your environment so you can use Emacs (v${_EMACS_VERSION})")

local apps_path = "${LOCATION}"

conflict("emacs")

prepend_path("PATH", pathJoin(apps_path, "/bin"))
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# Emacs end
################################################################


################################################################
# GCC Begin
################################################################
cd $dep_dir
_GCC_VERSION=11.4.0
LOCATION=$dep_dir/gcc/${_GCC_VERSION}
if [ -d ${LOCATION} ]; then
    echo "GCC version ${_GCC_VERSION} already installed"
else
    wget http://mirror.rit.edu/gnu/gcc/gcc-${_GCC_VERSION}/gcc-${_GCC_VERSION}.tar.gz
    tar xf gcc-${_GCC_VERSION}.tar.gz
    cd gcc-${_GCC_VERSION}
    # download_prerequisites gets things like mpc, mpfr, and gmp
    ./contrib/download_prerequisites
    ./configure --disable-multilib --enable-languages=c,c++,fortran \
                --enable-lto \
                --enable-quad \
                --with-system-zlib \
                --enable-host-shared \
                --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make -j${PARALLEL_MAKE_ARG} install
    cd $dep_dir
    rm -rf gcc-${_GCC_VERSION} gcc-${_GCC_VERSION}.tar.gz
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=$dep_dir/modules/global/gcc/${_GCC_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
# We install module files into their own sub-directory so that all packages
# built with the same GCC version are in the same sub-directory.
MODULE_GCC_LOCATION=$dep_dir/modules/gcc-${_GCC_VERSION}
mkdir -p ${MODULE_GCC_LOCATION}

if [ -f ${MODULE_FILE} ]; then
    echo "Module already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tSets GCC_HOME to be \$apps_path"
        puts stderr "\tDefines CC, CXX, F77 and F90 to be GNU compilers"
        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib64 to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib64 to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/lib64 to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/share/man to MANPATH"
        puts stderr "\tenvironment variables."
        puts stderr ""
        puts stderr "\tThis allows you to use the GNU Compilers (v${_GCC_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment to use the GNU compilers (v${_GCC_VERSION})"

# for Tcl script use only
set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        gcc intel pgi open64 gnu gnu12

pushenv  GCC_VERSION     $_GCC_VERSION
pushenv  GCC_HOME        \$apps_path
pushenv  CC              gcc
pushenv  CXX             g++
pushenv  F77             gfortran
pushenv  F90             gfortran

prepend-path    PATH            \$apps_path/bin
prepend-path    CPATH           \$apps_path/include
prepend-path    C_INCLUDE_PATH           \$apps_path/include
prepend-path    INCLUDE           \$apps_path/include
prepend-path    LIBRARY_PATH    \$apps_path/lib64
prepend-path    LD_RUN_PATH     \$apps_path/lib64
prepend-path    LD_LIBRARY_PATH \$apps_path/lib64
prepend-path    MANPATH         \$apps_path/share/man
prepend-path  CMAKE_PREFIX_PATH  \$apps_path
prepend-path MODULEPATH ${MODULE_GCC_LOCATION}/libraries
prepend-path MODULEPATH ${MODULE_GCC_LOCATION}/tools
EOF
    chmod -R 555 ${MODULE_FILE}
fi

module load gcc/${_GCC_VERSION}
INSTALL_GCC_LOCATION=$dep_dir/gcc-${_GCC_VERSION}
mkdir -p ${INSTALL_GCC_LOCATION}
################################################################
# GCC Begin
################################################################

################################################################
# hdf5 Begin
################################################################
cd $dep_dir
_HDF5_VERSION=1.12.3
LOCATION=${INSTALL_GCC_LOCATION}/hdf5/${_HDF5_VERSION}
if [ -d ${LOCATION} ]; then
    echo "HDF5 version ${_HDF5_VERSION} already installed"
else
    wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_${_HDF5_VERSION//\./_}/src/hdf5-${_HDF5_VERSION}.tar.gz
    tar -xzf hdf5-${_HDF5_VERSION}.tar.gz && cd hdf5-${_HDF5_VERSION}
    mkdir -p ${LOCATION}
    ./configure CC=gcc --enable-build-mode=production \
                --enable-file-locking=yes --with-pic \
                --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG} && make install PREFIX=${LOCATION}
    cd ..
    rm -rf hdf5-${_HDF5_VERSION} hdf5-${_HDF5_VERSION}.tar.gz
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/hdf5/${_HDF5_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## HDF5 built without file locking
##
proc ModulesHelp { } {
  global dotversion
  global apps_path

  puts stderr "\Sets HDF5_HOME to be \$apps_path"
  puts stderr "\Sets HDF5_ROOT to be \$apps_path"
  puts stderr "\tPrepends \$apps_path/bin to PATH"
  puts stderr "\tPrepends \$apps_path/include to CPATH"
  puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
  puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
  puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
  puts stderr "\tenvironment variables."
  puts stderr ""
  puts stderr "\tThis allows you to use HDF5 tools and libraries (v${_HDF5_VERSION})."
  puts stderr ""
}

module-whatis "Sets up your environment to use HDF5 (v${_HDF5_VERSION})"

# for Tcl script use only
set dotversion  3.2.6

set apps_path ${LOCATION}

conflict  HDF4
conflict  HDF5

setenv  HDF5_VERSION  ${_HDF5_VERSION}
setenv  HDF5_HOME \$apps_path
setenv  HDF5_ROOT \$apps_path

prepend-path  CMAKE_PREFIX_PATH \$apps_path/
prepend-path  PATH              \$apps_path/bin
prepend-path  CPATH             \$apps_path/include
prepend-path  LIBRARY_PATH      \$apps_path/lib
prepend-path  LD_RUN_PATH       \$apps_path/lib
prepend-path  LD_LIBRARY_PATH   \$apps_path/lib
EOF
    chmod -R 555 ${MODULE_FILE}
fi

module load hdf5/${_HDF5_VERSION}
################################################################
# hdf5 End
################################################################

################################################################
# Python Begin
################################################################
cd $dep_dir
_PYTHON_VERSION=3.10.3
PYTHON_INCLUDE_DIR=python3.10
LOCATION=${INSTALL_GCC_LOCATION}/python/${_PYTHON_VERSION}
_PYTHON_LOCATION=${INSTALL_GCC_LOCATION}/python/${_PYTHON_VERSION}

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/python/${_PYTHON_VERSION}.lua
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module already exists"
else
    # We set this up as a Lua module because that way we can use the depends_on
    # function to automatically load and unload HDF5. The TCL-based module files
    # seems to be the "old" way of doing them.
    cat >${MODULE_FILE} <<EOF
help([[
  Sets PYTHON_HOME to be ${LOCATION}
  Prepends ${LOCATION}/bin to PATH
  Prepends ${LOCATION}/include to CPATH
  Prepends ${LOCATION}/include/${PYTHON_INCLUDE_DIR} to CPATH
  Prepends ${LOCATION}/lib to LIBRARY_PATH
  Prepends ${LOCATION}/lib to LD_RUN_PATH
  Prepends ${LOCATION}/lib to LD_LIBRARY_PATH
  Prepends ${LOCATION}/share/man to MANPATH
  environment variables.

  This allows you to use the Python (v${_PYTHON_VERSION})
]])

whatis("Sets up your environment to use the Python (v${_PYTHON_VERSION})")

conflict("python")

depends_on("hdf5/${_HDF5_VERSION}")

pushenv("PYTHON_VERSION", "${_PYTHON_VERSION}")
pushenv("PYTHON_HOME", "${LOCATION}")

local apps_path = "${LOCATION}"

prepend_path("PATH", pathJoin(apps_path, "/bin"))
prepend_path("CPATH", pathJoin(apps_path, "/include"))
prepend_path("CPATH", pathJoin(apps_path, "/include/${PYTHON_INCLUDE_DIR}"))
prepend_path("C_INCLUDE_PATH", pathJoin(apps_path, "/include"))
prepend_path("C_INCLUDE_PATH", pathJoin(apps_path, "/include/${PYTHON_INCLUDE_DIR}"))
prepend_path("PKG_CONFIG_PATH", pathJoin(apps_path, "/lib/pkgconfig"))
prepend_path("LIBRARY_PATH", pathJoin(apps_path, "/lib"))
prepend_path("LD_RUN_PATH", pathJoin(apps_path, "/lib"))
prepend_path("LD_LIBRARY_PATH", pathJoin(apps_path, "/lib"))
prepend_path("MANPATH", pathJoin(apps_path, "/share/man"))
prepend_path("CMAKE_PREFIX_PATH", apps_path)
prepend_path("PYTHONPATH", pathJoin(apps_path, "/lib/${PYTHON_INCLUDE_DIR}/site-packages"))
EOF
    chmod -R 555 ${MODULE_FILE}
fi

module load python/${_PYTHON_VERSION}

cd $dep_dir
if [ -d ${LOCATION} ]; then
    echo "Python version ${_PYTHON_VERSION} already installed"
else
    wget https://github.com/libffi/libffi/archive/refs/tags/v3.4.4.tar.gz
    tar xf v3.4.4.tar.gz
    cd libffi-3.4.4
    ./autogen.sh
    ./configure --prefix=${LOCATION} --disable-docs
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd $dep_dir
    rm -rf libffi-3.4.4 v3.4.4.tar.gz
    # Some versions of python, like 3.10.3, don't search lib64, even if it's in
    # the paths, so symlink libffi to the lib directory. We don't add lib64
    # to the PATHs so other projects don't find them multiple times.
    mkdir -p ${LOCATION}/lib/
    cd ${LOCATION}/lib/
    ln -s ../lib64/* ./
    cd $dep_dir

    # Build ncurses.
    #
    # We need to include termlib and termcap, and also to symlink a lot of
    # headers so that python picks them up.
    wget https://invisible-island.net/datafiles/release/ncurses.tar.gz
    tar xf ncurses.tar.gz
    cd ./ncurses-6.3
    ./configure --with-libtool --with-termlib --with-shared \
                --with-gpm --enable-termcap --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make install
    ln -s ${LOCATION}/include/ncurses/ncurses.h ${LOCATION}/include/ncurses.h
    ln -s ${LOCATION}/include/ncurses/ncurses_dll.h \
       ${LOCATION}/include/ncurses_dll.h
    ln -s ${LOCATION}/include/ncurses/term.h ${LOCATION}/include/term.h
    ln -s ${LOCATION}/include/ncurses/curses.h ${LOCATION}/include/curses.h
    ln -s ${LOCATION}/include/ncurses/panel.h ${LOCATION}/include/panel.h
    cd ../
    rm -rf ncurses-6.3 ncurses.tar.gz

    # Build readline.
    #
    # To avoid undefined symbol UP, we need to build against termcap.
    wget https://git.savannah.gnu.org/cgit/readline.git/snapshot/readline-8.2.tar.gz
    tar xf readline-8.2.tar.gz
    cd readline-8.2
    ./configure --disable-install-examples --with-shared --enable-shared \
                --with-shared-termcap-library --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf readline-8.2 readline-8.2.tar.gz

    # Add gdmb3 just in case somebody needs it.
    wget https://ftp.gnu.org/gnu/gdbm/gdbm-1.23.tar.gz
    tar xf gdbm-1.23.tar.gz
    cd gdbm-1.23
    ./configure --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf gdbm-1.23 gdbm-1.23.tar.gz

    # Build cpython (python itself).
    #
    # Other dependencies like gdbm, dbm, uuid, and tcl+tk (for tkinter) could be
    # compiled before python.
    wget https://github.com/python/cpython/archive/refs/tags/v$_PYTHON_VERSION.tar.gz
    tar xf v${_PYTHON_VERSION}.tar.gz
    cd cpython-${_PYTHON_VERSION}
    ./configure --enable-optimizations --with-lto --enable-shared \
                --with-curses --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make install
    # Create links for python and pip
    if [ ! -e ${LOCATION}/bin/python ]; then
        ln -s ${LOCATION}/bin/python3 ${LOCATION}/bin/python
    fi
    if [ ! -e ${LOCATION}/bin/pip ]; then
        ln -s ${LOCATION}/bin/pip3 ${LOCATION}/bin/pip
    fi
    cd ../
    rm -rf ./cpython-${_PYTHON_VERSION} v${_PYTHON_VERSION}.tar.gz

    # Update pip to latest version
    pip install -U pip
    # Install meson, mako, numpy, and h5py to use our compiled HDF5.
    pip3 install meson mako numpy cmake-language-server
    HDF5_DIR=${LOCATION} pip3 install --no-binary=h5py --no-cache-dir h5py

    chmod -R 555 ${LOCATION}
fi
################################################################
# Python End
################################################################

################################################################
# CMake Begin
################################################################
cd $dep_dir
_CMAKE_VERSION=3.28.1
LOCATION=${INSTALL_GCC_LOCATION}/cmake/${_CMAKE_VERSION}
if [ -d ${LOCATION} ]; then
    echo "CMake version ${_CMAKE_VERSION} already installed"
else
    wget -O cmake-install.sh \
         "https://github.com/Kitware/CMake/releases/download/v${_CMAKE_VERSION}/cmake-${_CMAKE_VERSION}-linux-x86_64.sh"
    mkdir -p ${LOCATION}
    sh cmake-install.sh --prefix=${LOCATION} --skip-license
    rm cmake-install.sh
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/cmake/${_CMAKE_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tSets CMAKE_HOME to be \$apps_path"
        puts stderr "\tPrepends $apps_path/bin to PATH"
        puts stderr "\tenvironment variable."
        puts stderr ""
        puts stderr "\tThis allows you to run cmake (${_CMAKE_VERSION})"
        puts stderr ""
}

module-whatis   "Allows users to run cmake (${_CMAKE_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        cmake

pushenv CMAKE_VERSION ${_CMAKE_VERSION}
pushenv CMAKE_HOME \$apps_path

prepend-path    PATH            \$apps_path/bin
EOF
    chmod -R 555 ${MODULE_FILE}
fi

module load cmake/${_CMAKE_VERSION}
################################################################
# CMake End
################################################################

################################################################
# Ninja Begin
################################################################
cd $dep_dir
_NINJA_VERSION=1.10.1
LOCATION=${INSTALL_GCC_LOCATION}/ninja/${_NINJA_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Ninja version ${_NINJA_VERSION} already installed"
else
    wget https://github.com/ninja-build/ninja/archive/refs/tags/v${_NINJA_VERSION}.tar.gz \
         -O ninja.tar.gz
    tar -xzf ninja.tar.gz && cd ninja-${_NINJA_VERSION}
    mkdir -p ${LOCATION}
    mkdir build
    cd build
    cmake -D CMAKE_INSTALL_PREFIX=${LOCATION} \
          -D CMAKE_BUILD_TYPE=Release ..
    make -j${PARALLEL_MAKE_ARG} install
    cd $dep_dir
    rm -rf ninja.tar.gz ninja-${_NINJA_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/ninja/${_NINJA_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tSets NINJA_HOME to be \$apps_path"
        puts stderr "\tPrepends $apps_path/bin to PATH"
        puts stderr "\tenvironment variable."
        puts stderr ""
        puts stderr "\tThis allows you to run ninja (${_NINJA_VERSION})"
        puts stderr ""
}

module-whatis   "Allows users to run ninja (${_NINJA_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        ninja

setenv NINJA_VERSION ${_NINJA_VERSION}
setenv NINJA_HOME \$apps_path

prepend-path    PATH            \$apps_path/bin
EOF
    chmod -R 555 ${MODULE_FILE}
fi

module load ninja/${_NINJA_VERSION}
################################################################
# Ninja End
################################################################

################################################################
# hwloc Begin
################################################################
cd $dep_dir
_HWLOC_VERSION=2.10.0
LOCATION=${INSTALL_GCC_LOCATION}/hwloc/${_HWLOC_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Hwloc ${_HWLOC_VERSION} is already installed"
else
    wget https://download.open-mpi.org/release/hwloc/v2.10/hwloc-${_HWLOC_VERSION}.tar.gz \
         -O hwloc.tar.gz
    tar -xzf hwloc.tar.gz
    cd hwloc-${_HWLOC_VERSION}
    ./configure --disable-cuda --with-pick \
                --prefix=${LOCATION}
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ..
    rm -r hwloc.tar.gz hwloc-${_HWLOC_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/hwloc/${_HWLOC_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tSets HWLOC_HOME to be \$apps_path"
        puts stderr "\tPrepends $apps_path/bin to PATH"
        puts stderr "\tenvironment variable."
        puts stderr ""
        puts stderr "\tThis allows you to run hwloc (${_HWLOC_VERSION})"
        puts stderr ""
}

module-whatis   "Allows users to run hwloc (${_HWLOC_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        hwloc

setenv HWLOC_VERSION ${_HWLOC_VERSION}
setenv HWLOC_HOME \$apps_path

prepend-path    PATH            \$apps_path/bin
prepend-path    LIBRARY_PATH    \$apps_path/lib
prepend-path    LD_RUN_PATH     \$apps_path/lib
prepend-path    LD_LIBRARY_PATH \$apps_path/lib
prepend-path    MANPATH         \$apps_path/share/man
EOF
    chmod -R 555 ${MODULE_FILE}
fi

module load hwloc/${_HWLOC_VERSION}
################################################################
# hwloc end
################################################################

################################################################
# OpenBLAS Begin
################################################################
cd $dep_dir
_OPENBLAS_VERSION=0.3.25
LOCATION=${INSTALL_GCC_LOCATION}/openblas/${_OPENBLAS_VERSION}
if [ -d ${LOCATION} ]; then
    echo "OpenBLAS ${_OPENBLAS_VERSION} is already installed"
else
    wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v${_OPENBLAS_VERSION}/OpenBLAS-${_OPENBLAS_VERSION}.tar.gz
    tar -xzf OpenBLAS-${_OPENBLAS_VERSION}.tar.gz
    cd OpenBLAS-${_OPENBLAS_VERSION}
    make FC=gfortran USE_THREAD=0 USE_LOCKING=1 NO_AVX512=1 -j${PARALLEL_MAKE_ARG}
    mkdir -p ${LOCATION}
    make PREFIX=${LOCATION} install
    cd ..
    rm -r OpenBLAS-${_OPENBLAS_VERSION}.tar.gz OpenBLAS-${_OPENBLAS_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/openblas/${_OPENBLAS_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## OpenBLAS v${_OPENBLAS_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use OpenBLAS (v${_OPENBLAS_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use the OpenBLAS (v${_OPENBLAS_VERSION}) libraries"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        openblas

setenv  BLAS_VERSION    0.3.25
setenv  BLAS_HOME       \$apps_path

prepend-path    LIBRARY_PATH         \$apps_path/lib
prepend-path    LD_LIBRARY_PATH    \$apps_path/lib
prepend-path    CPATH                      \$apps_path/include
prepend-path  CMAKE_PREFIX_PATH  \$apps_path
prepend-path  PATH               \$apps_path/bin
EOF
    chmod -R 555 ${MODULE_FILE}
fi

module load openblas/${_OPENBLAS_VERSION}
################################################################
# OpenBLAS End
################################################################

################################################################
# libxsmm Begin
################################################################
cd $dep_dir
_LIBXSMM_VERSION=1.16.1
LOCATION=${INSTALL_GCC_LOCATION}/libxsmm/${_LIBXSMM_VERSION}
if [ -d ${LOCATION} ]; then
    echo "LIBXSMM ${_LIBXSMM_VERSION} already installed."
else
    wget https://github.com/hfp/libxsmm/archive/${_LIBXSMM_VERSION}.tar.gz -O libxsmm.tar.gz
    tar -xzf libxsmm.tar.gz && rm libxsmm.tar.gz && cd libxsmm-*
    make AVX=2 PREFIX=${LOCATION} install -j${PARALLEL_MAKE_ARG}
    cd ../
    rm -r libxsmm-*
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/libxsmm/${_LIBXSMM_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## LIBXSMM for small matrix multiplications ${_LIBXSMM_VERSION}
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use libxsmm (${_LIBXSMM_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use libxsmm (${_LIBXSMM_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        libxsmm

prepend-path CPATH              \$apps_path/include
prepend-path LIBRARY_PATH       \$apps_path/lib
prepend-path LD_LIBRARY_PATH    \$apps_path/lib
prepend-path LD_RUN_PATH        \$apps_path/lib
prepend-path CMAKE_PREFIX_PATH  \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi

module load libxsmm/${_LIBXSMM_VERSION}
################################################################
# libxsmm End
################################################################

################################################################
# gsl Begin
################################################################
cd $dep_dir
_GSL_VERSION=2.7
LOCATION=${INSTALL_GCC_LOCATION}/gsl/${_GSL_VERSION}
if [ -d ${LOCATION} ]; then
    echo "GSL ${_GSL_VERSION} already installed"
else
    wget https://mirror.ibcp.fr/pub/gnu/gsl/gsl-${_GSL_VERSION}.tar.gz
    tar -xzf gsl-${_GSL_VERSION}.tar.gz && cd gsl-${_GSL_VERSION}
    ./configure --prefix=${LOCATION} --with-pic
    make -j${PARALLEL_MAKE_ARG} && make install
    cd ..
    rm -rf ./gsl-${_GSL_VERSION}.tar.gz ./gsl-${_GSL_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/gsl/${_GSL_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## GNU GSL ${_GSL_VERSION}
proc ModulesHelp { } {
        global dotversion
        global apps_path

  puts stderr "\tSets env variable _GSL_VERSIONN to ${_GSL_VERSION}"
  puts stderr "\tSets env variable GSL_HOME to \$apps_path"
        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr "\tPrepends \$apps_path/share/man to MANPATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use GSL (v${_GSL_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use GSL (v${_GSL_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        gsl

setenv GSL_VERSION "${_GSL_VERSION}"
setenv GSL_HOME \$apps_path

prepend-path CPATH              \$apps_path/include
prepend-path LIBRARY_PATH       \$apps_path/lib
prepend-path LD_LIBRARY_PATH    \$apps_path/lib
prepend-path LD_RUN_PATH        \$apps_path/lib
prepend-path CMAKE_PREFIX_PATH  \$apps_path
prepend-path PATH               \$apps_path/bin
prepend-path MANPATH            \$apps_path/share/man
EOF
    chmod -R 555 ${MODULE_FILE}
fi

module load gsl/${_GSL_VERSION}
################################################################
# gsl End
################################################################

################################################################
# Blaze Begin
################################################################
cd $dep_dir
_BLAZE_VERSION=3.8
LOCATION=${INSTALL_GCC_LOCATION}/blaze/${_BLAZE_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Blaze version ${_BLAZE_VERSION} already installed"
else
    wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-${_BLAZE_VERSION}.tar.gz -O blaze.tar.gz
    tar -xzf blaze.tar.gz && cd blaze-${_BLAZE_VERSION}
    cmake -D CMAKE_INSTALL_PREFIX=${LOCATION} \
          -D CMAKE_BUILD_TYPE=Release .
    make install
    cd ../
    rm -rf ./blaze.tar.gz ./blaze-${_BLAZE_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/blaze/${_BLAZE_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## Blaze v${_BLAZE_VERSION}
##
## Install using Install.sh
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use header-only library Blaze (v3.8)"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use the header-only library Blaze (v3.8)"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        blaze

prepend-path    LIBRARY_PATH        \$apps_path/lib
prepend-path    LD_LIBRARY_PATH   \$apps_path/lib
prepend-path    CPATH                     \$apps_path/include
prepend-path  CMAKE_PREFIX_PATH \$apps_path/
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# Blaze End
################################################################

################################################################
# Boost Begin
################################################################
cd $dep_dir
_BOOST_VERSION=1.82.0
LOCATION=${INSTALL_GCC_LOCATION}/boost/${_BOOST_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Boost version ${_BOOST_VERSION} already installed"
else
    wget https://boostorg.jfrog.io/artifactory/main/release/${_BOOST_VERSION}/source/boost_${_BOOST_VERSION//\./_}.tar.gz -O boost.tar.gz
    tar -xzf boost.tar.gz && cd boost_${_BOOST_VERSION//\./_}
    mkdir -p ${LOCATION}
    ./bootstrap.sh --prefix=${LOCATION}
    ./b2 --without-python --without-mpi install
    cd ..
    rm -rf boost.tar.gz boost_${_BOOST_VERSION//\./_}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/boost/${_BOOST_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## Boost ${_BOOST_VERSION}
proc ModulesHelp { } {
        global dotversion
        global apps_path

  puts stderr "\tSets env variable _BOOST_VERSIONN to ${_BOOST_VERSION}"
  puts stderr "\tSets env variable BOOST_HOME to \$apps_path"
        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use Boost (v${_BOOST_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use Boost (v${_BOOST_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        boost

setenv BOOST_VERSION "${_BOOST_VERSION}"
setenv BOOST_HOME \$apps_path

prepend-path CPATH              \$apps_path/include
prepend-path LIBRARY_PATH       \$apps_path/lib
prepend-path LD_LIBRARY_PATH    \$apps_path/lib
prepend-path LD_RUN_PATH        \$apps_path/lib
prepend-path CMAKE_PREFIX_PATH  \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# Boost End
################################################################

################################################################
# Brigand Begin
################################################################
cd $dep_dir
_BRIGAND_VERSION=`date +%Y.%m.%d`
LOCATION=${INSTALL_GCC_LOCATION}/brigand/${_BRIGAND_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Brigand version ${_BRIGAND_VERSION} already installed"
else
    git clone https://github.com/edouarda/brigand.git brigand_${_BRIGAND_VERSION}
    cd brigand_${_BRIGAND_VERSION}
    mkdir -p ${LOCATION}
    cp -r include/ ${LOCATION}
    cd ..
    rm -rf brigand_${_BRIGAND_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/brigand/${_BRIGAND_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
## brigand/master
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use header-only template metaprogramming library Brigand"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use the header-only template metaprogramming library Brigand at it's master branch of ${_BRIGAND_VERSION}"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        brigand

prepend-path CPATH              \$apps_path/include
prepend-path CMAKE_PREFIX_PATH  \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# Brigand End
################################################################

################################################################
# jemalloc Begin
################################################################
cd $dep_dir
_JEMALLOC_VERSION=5.3.0
LOCATION=${INSTALL_GCC_LOCATION}/jemalloc/${_JEMALLOC_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Jemalloc version ${_JEMALLOC_VERSION} already installed"
else
    wget https://github.com/jemalloc/jemalloc/archive/refs/tags/${_JEMALLOC_VERSION}.tar.gz
    tar -xzf ${_JEMALLOC_VERSION}.tar.gz
    cd jemalloc-${_JEMALLOC_VERSION}
    mkdir -p ${LOCATION}
    ./autogen.sh --prefix=${LOCATION}
    make CFLAGS=-fPIC -j${PARALLEL_MAKE_ARG}
    make install
    cd ..
    rm -rf ${_JEMALLOC_VERSION}.tar.gz jemalloc-${_JEMALLOC_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/jemalloc/${_JEMALLOC_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## jemalloc ${_JEMALLOC_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr "\tPrepends \$apps_path/share/man to MANPATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use jemalloc (v${_JEMALLOC_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use jemalloc (v${_JEMALLOC_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        jemalloc

prepend-path CPATH              \$apps_path/include
prepend-path LIBRARY_PATH       \$apps_path/lib
prepend-path LD_LIBRARY_PATH    \$apps_path/lib
prepend-path LD_RUN_PATH        \$apps_path/lib
prepend-path CMAKE_PREFIX_PATH  \$apps_path
prepend-path PATH               \$apps_path/bin
prepend-path MANPATH            \$apps_path/share/man
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# jemalloc End
################################################################

################################################################
# catch Begin
################################################################
cd $dep_dir
_CATCH_VERSION=3.5.1
LOCATION=${INSTALL_GCC_LOCATION}/catch/${_CATCH_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Catch version ${_CATCH_VERSION} already installed"
else
    wget https://github.com/catchorg/Catch2/archive/refs/tags/v${_CATCH_VERSION}.tar.gz -O catch.tar.gz
    tar -xzf catch.tar.gz && cd Catch2-${_CATCH_VERSION}
    mkdir -p ${LOCATION}
    cmake  -B build -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=OFF \
           -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
           -D CMAKE_INSTALL_PREFIX=${LOCATION}
    cmake --build build/ --target install
    cd $dep_dir
    rm -rf catch.tar.gz Catch2-${_CATCH_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/catch/${_CATCH_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## Testing library Catch2 v${_CATCH_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use the testing library Catch2 (v${_CATCH_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use the testing library Catch2 (v${_CATCH_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        catch catch2 Catch Catch2

prepend-path CPATH              \$apps_path/include
prepend-path LIBRARY_PATH       \$apps_path/lib
prepend-path LD_LIBRARY_PATH    \$apps_path/lib
prepend-path LD_RUN_PATH        \$apps_path/lib
prepend-path CMAKE_PREFIX_PATH  \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# catch End
################################################################

################################################################
# libsharp Begin
################################################################
cd $dep_dir
_LIBSHARP_VERSION=1.0.0
LOCATION=${INSTALL_GCC_LOCATION}/libsharp/${_LIBSHARP_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Libsharp version ${_LIBSHARP_VERSION} already installed"
else
    wget https://github.com/Libsharp/libsharp/archive/v${_LIBSHARP_VERSION}.tar.gz -O libsharp.tar.gz
    tar -xzf libsharp.tar.gz && cd libsharp-*
    mkdir -p ${LOCATION}
    autoconf
    ./configure --prefix=${LOCATION} --disable-openmp --enable-pic
    make PREFIX=${LOCATION}
    cd $dep_dir
    rm -rf libsharp.tar.gz libsharp-*
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/libsharp/${_LIBSHARP_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## Spherical harmonic library libsharp v${_LIBSHARP_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use libsharp (v${_LIBSHARP_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment to use libsharp (v${_LIBSHARP_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        libsharp

prepend-path CPATH              \$apps_path/include
prepend-path LIBRARY_PATH       \$apps_path/lib
prepend-path LD_LIBRARY_PATH    \$apps_path/lib
prepend-path LD_RUN_PATH        \$apps_path/lib
prepend-path CMAKE_PREFIX_PATH  \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# libsharp End
################################################################

################################################################
# yaml-cpp Begin
################################################################
cd $dep_dir
_YAMLCPP_VERSION=0.8.0
LOCATION=${INSTALL_GCC_LOCATION}/yaml-cpp/${_YAMLCPP_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Yaml-Cpp version ${_YAMLCPP_VERSION} already installed"
else
    wget https://github.com/jbeder/yaml-cpp/archive/refs/tags/${_YAMLCPP_VERSION}.tar.gz -O yaml-cpp.tar.gz
    tar -xzf yaml-cpp.tar.gz && cd yaml-cpp-${_YAMLCPP_VERSION}
    mkdir build && cd build
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D YAML_CPP_BUILD_TESTS=OFF \
          -D CMAKE_CXX_COMPILER=g++ \
          -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
          -D YAML_CPP_BUILD_CONTRIB=OFF \
          -D YAML_CPP_BUILD_TOOLS=ON \
          -D CMAKE_INSTALL_PREFIX=${LOCATION} ../
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd $dep_dir
    rm -rf yaml-cpp.tar.gz yaml-cpp-*
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/yaml-cpp/${_YAMLCPP_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## Spherical harmonic library yaml-cpp v${_YAMLCPP_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use yaml-cpp (v${_YAMLCPP_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment to use yaml-cpp (v${_YAMLCPP_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        yaml-cpp

prepend-path CPATH              \$apps_path/include
prepend-path LIBRARY_PATH       \$apps_path/lib
prepend-path LD_LIBRARY_PATH    \$apps_path/lib
prepend-path LD_RUN_PATH        \$apps_path/lib
prepend-path CMAKE_PREFIX_PATH  \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# yaml-cpp End
################################################################

################################################################
# Google Benchmark Begin
################################################################
cd $dep_dir
_GOOGLE_BENCHMARK_VERSION=1.8.3
LOCATION=${INSTALL_GCC_LOCATION}/google_benchmark/${_GOOGLE_BENCHMARK_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Google Benchmark version ${_GOOGLE_BENCHMARK_VERSION} already installed"
else
    wget https://github.com/google/benchmark/archive/refs/tags/v${_GOOGLE_BENCHMARK_VERSION}.tar.gz -O google_benchmark.tar.gz
    tar -xzf google_benchmark.tar.gz && cd benchmark-${_GOOGLE_BENCHMARK_VERSION}
    mkdir build && cd build
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
          -D BENCHMARK_DOWNLOAD_DEPENDENCIES=OFF \
          -D BENCHMARK_ENABLE_TESTING=OFF \
          -D CMAKE_INSTALL_PREFIX=${LOCATION} ../
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd $dep_dir
    rm -rf google_benchmark.tar.gz benchmark-*
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/google_benchmark/${_GOOGLE_BENCHMARK_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## Spherical harmonic library google_benchmark v${_GOOGLE_BENCHMARK_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use google_benchmark (v${_GOOGLE_BENCHMARK_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment to use google_benchmark (v${_GOOGLE_BENCHMARK_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        google_benchmark

prepend-path CPATH              \$apps_path/include
prepend-path LIBRARY_PATH       \$apps_path/lib
prepend-path LD_LIBRARY_PATH    \$apps_path/lib
prepend-path LD_RUN_PATH        \$apps_path/lib
prepend-path CMAKE_PREFIX_PATH  \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# Google Benchmark End
################################################################

################################################################
# LLVM Begin
################################################################
cd $dep_dir
_LLVM_VERSION=17.0.6
LOCATION=${INSTALL_GCC_LOCATION}/llvm/${_LLVM_VERSION}
if [ -d ${LOCATION} ]; then
    echo "LLVM version ${_LLVM_VERSION} already installed"
else
    git clone --depth=1 --branch llvmorg-${_LLVM_VERSION} \
        https://github.com/llvm/llvm-project.git
    mkdir -p llvm-project/build
    cd llvm-project/build

    GCC_DIR=$(dirname $(dirname `which gcc`))
    cmake -S ../llvm -B ./ \
          -D LLVM_ENABLE_PROJECTS="clang;clang-tools-extra;compiler-rt;lldb;lld" \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=${LOCATION} \
          -D LLVM_BUILD_LLVM_DYLIB=ON \
          -D ENABLE_LLVM_SHARED=ON \
          -D CMAKE_CXX_LINK_FLAGS="-Wl,-rpath,${GCC_DIR}/lib64 -L${GCC_DIR}/lib64" \
          ..
    make -j${PARALLEL_MAKE_ARG}
    make -j${PARALLEL_MAKE_ARG} install

    LLVM_DIR=${LOCATION}
    cd $LLVM_DIR/bin && ln -s $GCC_DIR/bin/* .
    cd $LLVM_DIR/include && ln -s $GCC_DIR/include/* .
    cd $LLVM_DIR/lib && ln -s $GCC_DIR/lib/* .
    if [ -d $LLVM_DIR/lib64 ]; then
        cd $LLVM_DIR/lib64 && ln -s $GCC_DIR/lib64/* .
    fi
    cd $LLVM_DIR/libexec && ln -s $GCC_DIR/libexec/* .
    cd $dep_dir
    rm -rf llvm-project
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/llvm/${_LLVM_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## LLVM v${_LLVM_VERSION}
##
## To check that the right GCC version is used, run clang++ -v. You will get
## something like:
##
## Found candidate GCC installation:
##    /opt/sxs/test/LLVM/17.0.6/bin/../lib/gcc/x86_64-pc-linux-gnu/12.2.0
## Selected GCC installation:
##    /opt/sxs/test/LLVM/17.0.6/bin/../lib/gcc/x86_64-pc-linux-gnu/12.2.0
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tThe LLVM project, including extras like clang-tidy,"
        puts stderr "\tclang-format, libc++, lld, and even templight."
        puts stderr "\tSets LLVM_HOME to be \$apps_path"
        puts stderr "\tDoes NOT define CC, and CXX to be LLVM clang compiler"
        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/share/man to MANPATH"
        puts stderr "\tenvironment variables."
        puts stderr ""
        puts stderr "\tAllows you to use the Clang Compiler (v${_LLVM_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment to use LLVM (v${_LLVM_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        llvm intel pgi open64

setenv  LLVM_VERSION    ${_LLVM_VERSION}
setenv  LLVM_HOME       \$apps_path
# Don't set the compilers so we can easily use clang tools like clang-format
# with GCC. Really, whenever someone compiles they should explicitly choose
# the compiler to use.
# setenv  CC              clang
# setenv  CXX             clang++

prepend-path    PATH            \$apps_path/bin
prepend-path    CPATH           \$apps_path/include
prepend-path    C_INCLUDE_PATH  \$apps_path/include
prepend-path    LIBRARY_PATH    \$apps_path/lib
prepend-path    LD_RUN_PATH     \$apps_path/lib
prepend-path    LD_LIBRARY_PATH \$apps_path/lib
prepend-path    MANPATH         \$apps_path/share/man
prepend-path    CMAKE_PREFIX_PATH               \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# LLVM End
################################################################

################################################################
# xsimd Begin
################################################################
cd $dep_dir
_XSIMD_VERSION=12.1.1
LOCATION=${INSTALL_GCC_LOCATION}/xsimd/${_XSIMD_VERSION}
if [ -d ${LOCATION} ]; then
    echo "xsimd version ${_XSIMD_VERSION} already installed"
else
    wget https://github.com/xtensor-stack/xsimd/archive/refs/tags/${_XSIMD_VERSION}.tar.gz \
         -O xsimd.tar.gz
    tar -xzf xsimd.tar.gz && cd xsimd-${_XSIMD_VERSION}
    cmake -D CMAKE_INSTALL_PREFIX=${LOCATION} \
          -D CMAKE_BUILD_TYPE=Release .
    make install
    cd ../
    rm -rf ./xsimd.tar.gz ./xsimd-${_XSIMD_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/xsimd/${_XSIMD_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## xsimd v${_XSIMD_VERSION}
##
## Install using Install.sh
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use header-only library xsimd (v${_XSIMD_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use the header-only library xsimd (v${_XSIMD_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        xsimd

prepend-path    LIBRARY_PATH        \$apps_path/lib
prepend-path    LD_LIBRARY_PATH   \$apps_path/lib
prepend-path    CPATH                     \$apps_path/include
prepend-path  CMAKE_PREFIX_PATH \$apps_path/
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# xsimd End
################################################################

################################################################
# ccache Begin
################################################################
cd $dep_dir
_CCACHE_VERSION=4.9
LOCATION=${INSTALL_GCC_LOCATION}/ccache/${_CCACHE_VERSION}
if [ -d ${LOCATION} ]; then
    echo "ccache version ${_CCACHE_VERSION} already installed"
else
    wget https://github.com/ccache/ccache/releases/download/v${_CCACHE_VERSION}/ccache-${_CCACHE_VERSION}-linux-x86_64.tar.xz \
         -O ccache.tar.xz
    tar xf ccache.tar.xz
    mkdir -p ${LOCATION}/bin
    cp ./ccache-${_CCACHE_VERSION}-linux-x86_64/ccache ${LOCATION}/bin
    rm -r ./ccache.tar.xz \
       ./ccache-${_CCACHE_VERSION}-linux-x86_64
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/ccache/${_CCACHE_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## ccache v${_CCACHE_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use ccache (v${_CCACHE_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use ccache (v${_CCACHE_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        ccache

prepend-path    PATH        \$apps_path/bin
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# ccache End
################################################################

################################################################
# Intel MPI begin
################################################################
cd $dep_dir
_IMPI_VERSION=2021.11
LOCATION=${INSTALL_GCC_LOCATION}/intel/mpi/${_IMPI_VERSION}
if [ -d ${LOCATION} ]; then
    echo "Intel MPI version ${_IMPI_VERSION} already installed"
else
    mkdir ./intel_build
    cd ./intel_build
    # Note: likely the install and the URL need to be updated together.
    IMPI_INSTALL=l_mpi_oneapi_p_2021.11.0.49513_offline.sh
    wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/2c45ede0-623c-4c8e-9e09-bed27d70fa33/${IMPI_INSTALL}
    chmod +x ./${IMPI_INSTALL}
    ./${IMPI_INSTALL} -a --silent --install-dir ${LOCATION} --eula accept
    cd ../
    rm -rf ./intel_build
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/impi/${_IMPI_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -e ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    ln -s ${LOCATION}/mpi/${_IMPI_VERSION}/etc/modulefiles/mpi/${_IMPI_VERSION} \
       ${MODULE_FILE}
fi
################################################################
# Intel MPI end
################################################################

################################################################
# FFTW begin
################################################################
cd $dep_dir
FFTW_VERSION=3.3.10
LOCATION=${INSTALL_GCC_LOCATION}/fftw/${FFTW_VERSION}
if [ -d ${LOCATION} ]; then
    echo "FFTW version ${FFTW_VERSION} already installed"
else
    wget https://fftw.org/fftw-${FFTW_VERSION}.tar.gz
    tar xzf fftw-${FFTW_VERSION}.tar.gz
    cd fftw-${FFTW_VERSION}
    ./configure --prefix=${LOCATION} \
                --enable-avx2 \
                --enable-fma \
                --disable-mpi \
                --disable-threads \
                --disable-openmp \
                --with-pic
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf fftw-${FFTW_VERSION} fftw-${FFTW_VERSION}.tar.gz
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/fftw/${FFTW_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## FFTW v${FFTW_VERSION}
##
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tFFTW ${FFTW_VERSION}."
        puts stderr "\tSets FFTW_HOME to be \$apps_path"
        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/share/man to MANPATH"
        puts stderr "\tenvironment variables."
        puts stderr ""
}

module-whatis   "Sets up your environment to use FFTW (v${FFTW_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        fftw

pushenv  FFTW_VERSION    ${FFTW_VERSION}
pushenv  FFTW_HOME       \$apps_path

prepend-path    PATH            \$apps_path/bin
prepend-path    CPATH           \$apps_path/include
prepend-path    LIBRARY_PATH    \$apps_path/lib
prepend-path    LD_RUN_PATH     \$apps_path/lib
prepend-path    LD_LIBRARY_PATH \$apps_path/lib
prepend-path    MANPATH         \$apps_path/share/man
prepend-path    CMAKE_PREFIX_PATH               \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# FFTW end
################################################################

################################################################
# libbacktrace begin
################################################################
cd $dep_dir
LIBBACKTRACE_VERSION=1.0
LOCATION=${INSTALL_GCC_LOCATION}/libbacktrace/${LIBBACKTRACE_VERSION}
if [ -d ${LOCATION} ]; then
    echo "libbacktrace version ${LIBBACKTRACE_VERSION} already installed"
else
    git clone --depth=1 https://github.com/ianlancetaylor/libbacktrace.git \
        libbacktrace_clone
    cd libbacktrace_clone
    ./configure --prefix=${LOCATION} --with-pic
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./libbacktrace_clone
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/libbacktrace/${LIBBACKTRACE_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## libbacktrace ${LIBBACKTRACE_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tSets BACKTRACE_HOME to be \$apps_path"
        puts stderr "\tPrepends \$apps_path/include to CPATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use libbacktrace (${LIBBACKTRACE_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment to use libbacktrace (${LIBBACKTRACE_VERSION})"

# for Tcl script use only
set     dotversion      3.1.6

set     apps_path       ${LOCATION}

pushenv  BACKTRACE_HOME  \$apps_path

prepend-path    CPATH           \$apps_path/include
prepend-path    C_INCLUDE_PATH  \$apps_path/include
prepend-path    LIBRARY_PATH    \$apps_path/lib
prepend-path    LD_LIBRARY_PATH \$apps_path/lib
prepend-path    LD_RUN_PATH     \$apps_path/lib
prepend-path CMAKE_PREFIX_PATH \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# libbacktrace end
################################################################

################################################################
# doxygen begin
################################################################
cd $dep_dir
DOXYGEN_VERSION=1.10.0
LOCATION=${INSTALL_GCC_LOCATION}/doxygen/${DOXYGEN_VERSION}
if [ -d ${LOCATION} ]; then
    echo "doxygen version ${DOXYGEN_VERSION} already installed"
else
    module load llvm/${_LLVM_VERSION}
    wget https://github.com/doxygen/doxygen/archive/refs/tags/Release_${DOXYGEN_VERSION//\./_}.tar.gz \
         -O doxygen.tar.gz
    tar xf doxygen.tar.gz
    cd doxygen-Release_${DOXYGEN_VERSION//\./_}
    mkdir -p ./build
    cd build
    # We get runtime errors if building against libclang, unfortunately.
    cmake -D CMAKE_INSTALL_PREFIX=${LOCATION} \
          -D CMAKE_BUILD_TYPE=Release \
          -D use_libclang=OFF \
          ..
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd $dep_dir
    rm -rf doxygen-Release_${DOXYGEN_VERSION//\./_}
    rm ./doxygen.tar.gz
    module unload llvm/${_LLVM_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/doxygen/${DOXYGEN_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## doxygen v${DOXYGEN_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use doxygen (v${DOXYGEN_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use doxygen (v${DOXYGEN_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        doxygen

prepend-path    PATH        \$apps_path/bin
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# doxygen end
################################################################

################################################################
# lcov begin
################################################################
cd $dep_dir
LCOV_VERSION=2.0
LOCATION=${INSTALL_GCC_LOCATION}/lcov/${LCOV_VERSION}
if [ -d ${LOCATION} ]; then
    echo "lcov version ${LCOV_VERSION} already installed"
else
    wget https://github.com/linux-test-project/lcov/archive/refs/tags/v${LCOV_VERSION}.tar.gz \
         -O lcov.tar.gz
    tar xf lcov.tar.gz
    cd ./lcov-${LCOV_VERSION}
    make PREFIX=${LOCATION} install
    cd $dep_dir
    rm -rf lcov.tar.gz ./lcov-${LCOV_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/lcov/${LCOV_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## lcov v${LCOV_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/share/man to MAN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use lcov (v${LCOV_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use lcov (v${LCOV_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        lcov

prepend-path    PATH        \$apps_path/bin
prepend-path    LIBRARY_PATH    \$apps_path/lib
prepend-path    LD_LIBRARY_PATH \$apps_path/lib
prepend-path    LD_RUN_PATH     \$apps_path/lib
prepend-path    MANPATH         \$apps_path/share/man
prepend-path CMAKE_PREFIX_PATH \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# lcov end
################################################################

################################################################
# petsc begin
################################################################
cd $dep_dir
PETSC_VERSION=3.13.6
LOCATION=${INSTALL_GCC_LOCATION}/petsc/${PETSC_VERSION}
if [ -d ${LOCATION} ]; then
    echo "petsc version ${PETSC_VERSION} already installed"
else
    module load \
           openblas/${_OPENBLAS_VERSION} \
           impi/${_IMPI_VERSION}
    git clone -b v${PETSC_VERSION} --depth=1 \
        https://gitlab.com/petsc/petsc.git petsc-${PETSC_VERSION}
    cd petsc-${PETSC_VERSION}
    ./configure \
        --prefix=${LOCATION} \
        --enable-debug=0 \
        --COPTFLAGS="-O3" --CXXOPTFLAGS="-O3" --FOPTFLAGS="-O3" \
        --with-blas-lib=$BLAS_HOME/lib/libopenblas.so \
        --with-lapack-lib=$BLAS_HOME/lib/libopenblas.so
    make -j${PARALLEL_MAKE_ARG}
    make install
    cd ../
    rm -rf ./petsc-${PETSC_VERSION}
    module unload \
           openblas/${_OPENBLAS_VERSION} \
           impi/${_IMPI_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/libraries/petsc/${PETSC_VERSION}
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
#%Module1.0#####################################################################
##
## dot modulefile
##
## petsc v${PETSC_VERSION}
##
proc ModulesHelp { } {
        global dotversion
        global apps_path

        puts stderr "\tPrepends \$apps_path/bin to PATH"
        puts stderr "\tPrepends \$apps_path/lib to LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_LIBRARY_PATH"
        puts stderr "\tPrepends \$apps_path/lib to LD_RUN_PATH"
        puts stderr "\tPrepends \$apps_path/share/man to MAN_PATH"
        puts stderr "\tPrepends \$apps_path/ to CMAKE_PREFIX_PATH"
        puts stderr ""
        puts stderr "\tSets up your environment to use petsc (v${PETSC_VERSION})"
        puts stderr ""
}

module-whatis   "Sets up your environment so you can use petsc (v${PETSC_VERSION})"

set     dotversion      3.2.6

set     apps_path       ${LOCATION}

conflict        petsc

setenv PETSC_HOME \$apps_path

prepend-path    PATH        \$apps_path/bin
prepend-path    LIBRARY_PATH    \$apps_path/lib
prepend-path    LD_LIBRARY_PATH \$apps_path/lib
prepend-path    LD_RUN_PATH     \$apps_path/lib
prepend-path    MANPATH         \$apps_path/share/man
prepend-path CMAKE_PREFIX_PATH \$apps_path
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# petsc end
################################################################

################################################################
# Charm++ begin
################################################################
cd $dep_dir
_CHARM_VERSION=7.0.0
LOCATION=${INSTALL_GCC_LOCATION}/charm/${_CHARM_VERSION}
if [ -d ${LOCATION} ]; then
    echo "charm version ${_CHARM_VERSION} already installed"
else
    module purge
    module load gcc/${_GCC_VERSION} \
           cmake/${_CMAKE_VERSION} \
           hwloc/${_HWLOC_VERSION} \
           impi/${_IMPI_VERSION}
    git clone git@github.com:UIUC-PPL/charm.git ${LOCATION}
    cd ${LOCATION}/
    git checkout v${_CHARM_VERSION}
    git apply $SPECTRE_HOME/support/Charm/v${_CHARM_VERSION}.patch
    # git add -u
    # git commit -m "Apply SpECTRE patch"
    ./build LIBS mpi-linux-x86_64 smp -g -fPIC --with-production \
            --destination=mpi-linux-x86_64-smp-gcc -j32
    cd $dep_dir
    chmod -R 555 ${LOCATION}

    git clone git@github.com:UIUC-PPL/charm.git ${LOCATION}-tracing
    cd ${LOCATION}-tracing/
    git checkout v${_CHARM_VERSION}
    git apply $SPECTRE_HOME/support/Charm/v${_CHARM_VERSION}.patch
    ./build LIBS mpi-linux-x86_64 smp -g -fPIC --with-production \
            --destination=mpi-linux-x86_64-smp-gcc-tracing --enable-tracing \
            -j32
    cd $dep_dir
    chmod -R 555 ${LOCATION}-tracing/

    module unload hwloc/${_HWLOC_VERSION} \
           impi/${_IMPI_VERSION}

fi

MODULE_FILE_BASE=${MODULE_GCC_LOCATION}/libraries/charm/${_CHARM_VERSION}
_write_charm_module() {
    MODULE_FILE=${MODULE_FILE_BASE}${2}.lua
    mkdir -p `dirname ${MODULE_FILE}`
    if [ -f ${MODULE_FILE} ]; then
        echo "Module already exists"
    else
        # We set this up as a Lua module because that way we can use the
        # depends_on function to automatically load and unload HDF5. The
        # TCL-based module files seems to be the "old" way of doing them.
        cat >${MODULE_FILE} <<EOF
help([[
  Sets CHARM_HOME to ${1}
  Sets CHARM_ROOT to ${1}
  Sets CHARM_VERSION to ${_CHARM_VERSION}
  Prepends ${1}/bin to PATH
  Prepends ${1}/include to CPATH
  Prepends ${1}/include to C_INCLUDE_PATH
  Prepends ${1}/lib to LIBRARY_PATH
  Prepends ${1}/lib to LD_RUN_PATH
  Prepends ${1}/lib to LD_LIBRARY_PATH
  Prepends ${1} to CMAKE_PREFIX_PATH
  environment variables.

  This allows you to use the Charm (v${_CHARM_VERSION})
]])

whatis("Sets up your environment to use the Charm (v${_CHARM_VERSION})")

conflict("charm")

local apps_path = "${1}"

depends_on("hwloc/${_HWLOC_VERSION}")
depends_on("impi/${_IMPI_VERSION}")

pushenv("CHARM_VERSION", "${_CHARM_VERSION}")
pushenv("CHARM_HOME", apps_path)
pushenv("CHARM_ROOT", apps_path)

prepend_path("PATH", pathJoin(apps_path, "/bin"))
prepend_path("CPATH", pathJoin(apps_path, "/include"))
prepend_path("C_INCLUDE_PATH", pathJoin(apps_path, "/include"))
prepend_path("LIBRARY_PATH", pathJoin(apps_path, "/lib"))
prepend_path("LD_RUN_PATH", pathJoin(apps_path, "/lib"))
prepend_path("LD_LIBRARY_PATH", pathJoin(apps_path, "/lib"))
prepend_path("CMAKE_PREFIX_PATH", apps_path)
EOF
        chmod -R 555 ${MODULE_FILE}
    fi
}
_write_charm_module "${LOCATION}/mpi-linux-x86_64-smp-gcc" ""
_write_charm_module "${LOCATION}-tracing/mpi-linux-x86_64-smp-gcc-tracing" "-tracing"
################################################################
# Charm++ end
################################################################

################################################################
# spectre-python begin
################################################################
cd $dep_dir
LOCATION=${INSTALL_GCC_LOCATION}/spectre-python/${_PYTHON_VERSION}
if [ -d ${LOCATION} ]; then
    echo "charm version ${_CHARM_VERSION} already installed"
else
    module load python/${_PYTHON_VERSION}
    python -m venv ${LOCATION}
    source ${LOCATION}/bin/activate
    python -m pip install -U pip
    python -m pip install --no-cache-dir \
        -r $SPECTRE_HOME/support/Python/requirements.txt \
        -r $SPECTRE_HOME/support/Python/dev_requirements.txt
    python -m pip install "pybind11[global]"
    # For paraview, etc.
    python -m pip install meson mako cmake-language-server
    deactivate
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/spectre-python/${_PYTHON_VERSION}.lua
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module already exists"
else
    # We set this up as a Lua module because that way we can use the depends_on
    # function to automatically load and unload HDF5. The TCL-based module files
    # seems to be the "old" way of doing them.
    cat >${MODULE_FILE} <<EOF
help([[
  This allows you to use the SpECTRE Python (v${_PYTHON_VERSION})
]])

whatis("Sets up your environment to use the SpECTRE Python (v${_PYTHON_VERSION})")

local apps_path = "${LOCATION}"

depends_on("hdf5/${_HDF5_VERSION}")
depends_on("python/${_PYTHON_VERSION}")

pushenv("VIRTUAL_ENV", "${LOCATION}")

prepend_path("PATH", pathJoin(apps_path, "/bin"))
prepend_path("PYTHONPATH", pathJoin(apps_path, "/lib/${PYTHON_INCLUDE_DIR}/site-packages"))
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# spectre-python end
################################################################

################################################################
# spectre-deps begin
################################################################
MODULE_FILE=${MODULE_GCC_LOCATION}/tools/spectre-deps.lua
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module already exists"
else
    # We set this up as a Lua module because that way we can use the depends_on
    # function to automatically load and unload HDF5. The TCL-based module files
    # seems to be the "old" way of doing them.
    cat >${MODULE_FILE} <<EOF
help([[
  Load SpECTRE dependencies.
]])

whatis("Loads SpECTRE dependencies")

conflict("spectre-deps")

depends_on("blaze/${_BLAZE_VERSION}")
depends_on("boost/${_BOOST_VERSION}")
depends_on("brigand/${_BRIGAND_VERSION}")
depends_on("catch/${_CATCH_VERSION}")
depends_on("charm/${_CHARM_VERSION}")
depends_on("fftw/${FFTW_VERSION}")
depends_on("google_benchmark/${_GOOGLE_BENCHMARK_VERSION}")
depends_on("gsl/${_GSL_VERSION}")
depends_on("hdf5/${_HDF5_VERSION}")
depends_on("impi/${_IMPI_VERSION}")
depends_on("jemalloc/${_JEMALLOC_VERSION}")
depends_on("libbacktrace/${LIBBACKTRACE_VERSION}")
depends_on("libsharp/${_LIBSHARP_VERSION}")
depends_on("libxsmm/${_LIBXSMM_VERSION}")
depends_on("openblas/${_OPENBLAS_VERSION}")
depends_on("petsc/${PETSC_VERSION}")
depends_on("xsimd/${_XSIMD_VERSION}")
depends_on("yaml-cpp/${_YAMLCPP_VERSION}")

depends_on("ccache/${_CCACHE_VERSION}")
depends_on("cmake/${_CMAKE_VERSION}")
depends_on("doxygen/${DOXYGEN_VERSION}")
depends_on("hwloc/${_HWLOC_VERSION}")
depends_on("lcov/${LCOV_VERSION}")
depends_on("llvm/${_LLVM_VERSION}")
depends_on("ninja/${_NINJA_VERSION}")
depends_on("spectre-python/${_PYTHON_VERSION}")
EOF
    chmod -R 555 ${MODULE_FILE}
fi
################################################################
# spectre-deps end
################################################################

################################################################
# paraview begin
################################################################
cd $dep_dir
MESA_VERSION=23.3.4
COMMON_LOCATION=${INSTALL_GCC_LOCATION}/paraview/common
if [ -d ${COMMON_LOCATION} ]; then
    echo "paraview commons version already installed"
else
    module load python/${_PYTHON_VERSION} \
           llvm/${_LLVM_VERSION} \
           impi/${_IMPI_VERSION} \
           hdf5/${_HDF5_VERSION} \
           libbacktrace/${LIBBACKTRACE_VERSION} \
           cmake/${_CMAKE_VERSION} \
           ninja/${_NINJA_VERSION} \
           llvm/${_LLVM_VERSION}
    # Set up mesa dependency libdrm
    wget https://gitlab.freedesktop.org/mesa/drm/-/archive/libdrm-2.4.120/drm-libdrm-2.4.120.tar.gz
    tar xf drm-libdrm-2.4.120.tar.gz
    cd drm-libdrm-2.4.120
    meson setup build -Dprefix=${COMMON_LOCATION}
    cd build
    ninja
    ninja install
    cd ../../
    rm -rf drm-libdrm-2.4.120.tar.gz drm-libdrm-2.4.120

    wget https://archive.mesa3d.org/mesa-${MESA_VERSION}.tar.xz
    tar xf mesa-${MESA_VERSION}.tar.xz
    cd mesa-${MESA_VERSION}
    PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${COMMON_LOCATION}/lib64/pkgconfig
    meson setup build/ -Dosmesa=true -Dgallium-drivers=swrast \
          -Dvulkan-drivers=swrast -Degl=disabled -Dglx=disabled \
          -Dgbm=disabled -Dopengl=true -Dgles1=disabled -Dgles2=disabled \
          -Dxmlconfig=disabled -Dexpat=disabled -Dcpp_rtti=false \
          -Dplatforms= -Dprefix=${COMMON_LOCATION}
    cd ./build
    ninja
    ninja install
    cd ../../
    rm -rf mesa-${MESA_VERSION} mesa-${MESA_VERSION}.tar.xz
    module unload python/${_PYTHON_VERSION} \
           llvm/${_LLVM_VERSION} \
           impi/${_IMPI_VERSION} \
           hdf5/${_HDF5_VERSION} \
           libbacktrace/${LIBBACKTRACE_VERSION} \
           cmake/${_CMAKE_VERSION} \
           ninja/${_NINJA_VERSION}
    chmod -R 555 ${LOCATION}
fi

MODULE_FILE=${MODULE_GCC_LOCATION}/tools/paraview/common.lua
mkdir -p `dirname ${MODULE_FILE}`
if [ -f ${MODULE_FILE} ]; then
    echo "Module file ${MODULE_FILE} already exists"
else
    cat >${MODULE_FILE} <<EOF
help([[
paraview dependencies
]])
whatis("Sets up ParaView dependencies")

local apps_path = "${COMMON_LOCATION}"

depends_on("boost/${_BOOST_VERSION}")
depends_on("llvm/${_LLVM_VERSION}")
depends_on("impi/${_IMPI_VERSION}")
depends_on("hdf5/${_HDF5_VERSION}")
depends_on("libbacktrace/${LIBBACKTRACE_VERSION}")
depends_on("cmake/${_CMAKE_VERSION}")
depends_on("ninja/${_NINJA_VERSION}")

prepend_path("CPATH", pathJoin(apps_path, "/include"))
prepend_path("C_INCLUDE_PATH", pathJoin(apps_path, "/include"))
prepend_path("LIBRARY_PATH", pathJoin(apps_path, "/lib64"))
prepend_path("LD_RUN_PATH", pathJoin(apps_path, "/lib64"))
prepend_path("LD_LIBRARY_PATH", pathJoin(apps_path, "/lib64"))
prepend_path("CMAKE_PREFIX_PATH", apps_path)
EOF
    chmod -R 555 ${MODULE_FILE}
fi

# Set up a few different versions of paraview so people can connect more
# easily
_build_paraview() {
    PARAVIEW_VERSION=$1
    LOCATION=${INSTALL_GCC_LOCATION}/paraview/${PARAVIEW_VERSION}
    if [ -d ${LOCATION} ]; then
        echo "paraview version ${PARAVIEW_VERSION} already installed"
    else
        module load python/${_PYTHON_VERSION} \
           paraview/common
        wget https://www.paraview.org/paraview-downloads/download.php\?submit\=Download\&version\=v${PARAVIEW_VERSION%.*}\&type\=source\&os\=Sources\&downloadFile\=ParaView-v${PARAVIEW_VERSION}.tar.xz \
             -O ParaView-v${PARAVIEW_VERSION}.tar.xz
        tar xf ParaView-v${PARAVIEW_VERSION}.tar.xz
        cd ParaView-v${PARAVIEW_VERSION}
        mkdir -p ./build
        cd ./build
        cmake -D CMAKE_INSTALL_PREFIX=${LOCATION} \
              -D CMAKE_BUILD_TYPE=Release \
              -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ \
              -D PARAVIEW_USE_QT=NO -D VTK_USE_X=OFF \
              -D OPENGL_INCLUDE_DIR=IGNORE -D OPENGL_gl_LIBRARY=IGNORE \
              -D VTK_OPENGL_HAS_OSMESA=ON \
              -D VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=ON \
              -D PARAVIEW_BUILD_SHARED_LIBS=ON -D BUILD_TESTING=OFF \
              -D VTK_SMP_IMPLEMENTATION_TYPE=OpenMP \
              -D PARAVIEW_ENABLE_XDMF2=ON -D PARAVIEW_ENABLE_XDMF3=ON \
              -D OSMESA_INCLUDE_DIR=${COMMON_LOCATION}/include \
              -D OSMESA_LIBRARY=${COMMON_LOCATION}/lib64/libOSMesa.so \
              -D CMAKE_INSTALL_RPATH=${COMMON_LOCATION}/lib64 \
              -D PARAVIEW_USE_PYTHON=YES \
              -Wno-dev \
              ..
        make -j${PARALLEL_MAKE_ARG}
        make install
        cd ../../
        rm -rf ./ParaView-v${PARAVIEW_VERSION} \
           ParaView-v${PARAVIEW_VERSION}.tar.xz
        module unload \
               python/${_PYTHON_VERSION} \
               paraview/common
        chmod -R 555 ${LOCATION}
    fi

    MODULE_FILE=${MODULE_GCC_LOCATION}/tools/paraview/${PARAVIEW_VERSION}.lua
    mkdir -p `dirname ${MODULE_FILE}`
    if [ -f ${MODULE_FILE} ]; then
        echo "Module file ${MODULE_FILE} already exists"
    else
        cat >${MODULE_FILE} <<EOF
help([[
  ParaView v${PARAVIEW_VERSION}
]])
whatis("Sets up your environment so you can use ParaView (v${PARAVIEW_VERSION})")

local apps_path = "${LOCATION}"

conflict("paraview")

depends_on("boost/${_BOOST_VERSION}")
depends_on("llvm/${_LLVM_VERSION}")
depends_on("impi/${_IMPI_VERSION}")
depends_on("hdf5/${_HDF5_VERSION}")

prepend_path("PATH", pathJoin(apps_path, "/bin"))
prepend_path("CPATH", pathJoin(apps_path, "/include"))
prepend_path("C_INCLUDE_PATH", pathJoin(apps_path, "/include"))
prepend_path("LIBRARY_PATH", pathJoin(apps_path, "/lib64"))
prepend_path("LD_RUN_PATH", pathJoin(apps_path, "/lib64"))
prepend_path("LD_LIBRARY_PATH", pathJoin(apps_path, "/lib64"))
prepend_path("CMAKE_PREFIX_PATH", apps_path)
EOF
        chmod -R 555 ${MODULE_FILE}
    fi
}
_build_paraview 5.10.1
_build_paraview 5.11.1
################################################################
# paraview end
################################################################
