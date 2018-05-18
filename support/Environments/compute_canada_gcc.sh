#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    local start_dir=`pwd`
    dep_dir=$1
    if [ $# != 1 ]; then
        echo "You must pass one argument to spectre_setup_modules, which"
        echo "is the directory where you want the dependencies to be built."
        return 1
    fi
    mkdir -p $dep_dir
    cd $dep_dir
    mkdir -p $dep_dir/modules

    if [[ $HOSTNAME =~ ^nia ]]; then
        module load CCEnv
        module load nixpkgs
    fi
    module load gcc/7.3.0
    export CMAKE_PREFIX_PATH=$EBROOTGCC:$CMAKE_PREFIX_PATH

    if [ -f catch/include/catch.hpp ]; then
        echo "Catch is already installed"
    else
        echo "Installing catch..."
        mkdir -p $dep_dir/catch/include
        cd $dep_dir/catch/include
        wget https://github.com/catchorg/Catch2/releases/download/v2.2.1/catch.hpp -O catch.hpp
        echo "Installed Catch into $dep_dir/catch"
        echo "#%Module1.0" > $dep_dir/modules/catch
        echo "prepend-path CPATH \"$dep_dir/catch/include\"" >> $dep_dir/modules/catch
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/catch/\"" >> $dep_dir/modules/catch
    fi
    cd $dep_dir

    if [ -f blaze/include/blaze/Blaze.h ]; then
        echo "Blaze is already installed"
    else
        echo "Installing Blaze..."
        mkdir -p $dep_dir/blaze/
        cd $dep_dir/blaze/
        wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.3.tar.gz -O blaze.tar.gz
        tar -xzf blaze.tar.gz
        mv blaze-* include
        echo "Installed Blaze into $dep_dir/blaze"
        echo "#%Module1.0" > $dep_dir/modules/blaze
        echo "prepend-path CPATH \"$dep_dir/blaze/include\"" >> $dep_dir/modules/blaze
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/blaze/\"" >> $dep_dir/modules/blaze
    fi
    cd $dep_dir


    if [ -f brigand/include/brigand/brigand.hpp ]; then
        echo "Brigand is already installed"
    else
        echo "Installing Brigand..."
        rm -rf $dep_dir/brigand
        git clone https://github.com/edouarda/brigand.git
        echo "Installed Brigand into $dep_dir/brigand"
        echo "#%Module1.0" > $dep_dir/modules/brigand
        echo "prepend-path CPATH \"$dep_dir/brigand/include\"" >> $dep_dir/modules/brigand
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/brigand/\"" >> $dep_dir/modules/brigand
    fi
    cd $dep_dir

    if [ -f $dep_dir/yaml-cpp/lib/libyaml-cpp.a ]; then
        echo "yaml-cpp is already installed"
    else
        echo "Installing yaml-cpp..."
        wget https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz -O yaml-cpp.tar.gz
        tar -xzf yaml-cpp.tar.gz
        mv yaml-cpp-* yaml-cpp-build
        cd $dep_dir/yaml-cpp-build
        mkdir build
        cd build
        cmake -D CMAKE_BUILD_TYPE=Release -D YAML_CPP_BUILD_TESTS=OFF \
              -D YAML_CPP_BUILD_CONTRIB=OFF \
              -D YAML_CPP_BUILD_TOOLS=ON \
              -D CMAKE_INSTALL_PREFIX=$dep_dir/yaml-cpp ..
        make -j4
        make install
        cd $dep_dir
        rm -r yaml-cpp-build
        rm -r yaml-cpp.tar.gz
        echo "Installed yaml-cpp into $dep_dir/yaml-cpp"
        echo "#%Module1.0" > $dep_dir/modules/yaml-cpp
        echo "prepend-path LIBRARY_PATH \"$dep_dir/yaml-cpp/lib\"" >> $dep_dir/modules/yaml-cpp
        echo "prepend-path LD_LIBRARY_PATH \"$dep_dir/yaml-cpp/lib\"" >> $dep_dir/modules/yaml-cpp
        echo "prepend-path CPATH \"$dep_dir/yaml-cpp/include\"" >> $dep_dir/modules/yaml-cpp
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/yaml-cpp/\"" >> $dep_dir/modules/yaml-cpp
    fi
    cd $dep_dir

    # Set up Charm++ because that can be difficult
    charm_target=mpi-linux-x86_64
    if [ -f $dep_dir/charm/$charm_target/lib/libck.a ]; then
        echo "Charm++ is already installed"
    else
        echo "Installing Charm++..."
        git clone https://charm.cs.illinois.edu/gerrit/charm
        cd $dep_dir/charm
        git checkout v6.8.2
        module load openmpi
        ./build charm++ $charm_target -j4 --with-production
        module unload openmpi
        git apply $SPECTRE_HOME/support/Charm/v6.8.patch
        cd $dep_dir
        rm charm.tar.gz
        echo "Installed Charm++ into $dep_dir/charm"
        echo "#%Module1.0" > $dep_dir/modules/charm
        echo "prepend-path LIBRARY_PATH \"$dep_dir/charm/lib\"" >> $dep_dir/modules/charm
        echo "prepend-path LD_LIBRARY_PATH \"$dep_dir/charm/lib\"" >> $dep_dir/modules/charm
        echo "prepend-path CPATH \"$dep_dir/charm/include\"" >> $dep_dir/modules/charm
        echo "prepend-path CMAKE_PREFIX_PATH \"$dep_dir/charm/\"" >> $dep_dir/modules/charm
        echo "setenv CHARM_VERSION 6.8.2" >> $dep_dir/modules/charm
        echo "setenv CHARM_HOME $dep_dir/charm/$charm_target" >> $dep_dir/modules/charm
        echo "setenv CHARM_ROOT $dep_dir/charm/$charm_target" >> $dep_dir/modules/charm
    fi
    cd $dep_dir

    if [ -f $dep_dir/modules/env_vars ]; then
        echo "Already set up environment variables"
    else
        echo "Setting up environment variables..."
        echo "#%Module1.0" > $dep_dir/modules/env_vars
        echo "prepend-path CMAKE_PREFIX_PATH \"$EBROOTGCC\"" >> $dep_dir/modules/env_vars
        echo "prepend-path CMAKE_PREFIX_PATH \"/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/libxsmm/1.8.2\"" >> $dep_dir/modules/env_vars
        echo "Done setting up environment "
    fi
    cd $start_dir

    module unload gcc/7.3.0

    printf "\n\nIMPORTANT!!!\nIn order to be able to use these modules you\n"
    echo "must run:"
    echo "  module use $dep_dir/modules"
    echo "You will need to do this every time you compile SpECTRE, so you may"
    echo "want to add it to your ~/.bashrc."
}

spectre_unload_modules() {
    module unload yaml-cpp
    module unload env_vars
    module unload charm
    module unload catch
    module unload brigand
    module unload blaze

    module unload scipy-stack/2017b
    module unload python/3.5.4
    module unload libxsmm/1.8.2
    module unload hdf5
    module unload gsl
    module unload boost
    module unload openmpi
    module unload gcc/7.3.0
    if [[ $HOSTNAME =~ ^nia ]]; then
        echo "Note: The nixpkgs and CCEnv are not unloaded by default"
    fi
}

spectre_load_modules() {
    echo "If you receive module not found errors make sure to run"
    echo "  module use /PATH/TO/DEPS/modules"
    echo "and then load the modules"
    if [[ $HOSTNAME =~ ^nia ]]; then
        module load CCEnv
        module load nixpkgs
    fi
    module load gcc/7.3.0
    module load openmpi
    module load boost
    module load gsl
    module load hdf5
    module load libxsmm/1.8.2
    module load python/3.5.4
    module load scipy-stack/2017b

    module load blaze
    module load brigand
    module load catch
    module load charm
    module load env_vars
    module load yaml-cpp
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          $SPECTRE_HOME
}
