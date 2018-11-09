#!/bin/bash -e

# Distributed under the MIT License.
# See LICENSE.txt for details.

if [ $SPECTRE_CONTAINER ]; then
    . /etc/profile.d/lmod.sh
    # Need to remove any existing modules in case the user had
    # those loaded in their bashrc and they started the shell using
    # `singularity exec spectre.img /bin/bash`
    module purge
    export PATH=/work/spack/bin:$PATH
    . /work/spack/share/spack/setup-env.sh
    spack load catch
    spack load brigand
    spack load blaze
    spack load gsl
    spack load libsharp
    spack load libxsmm
    spack load yaml-cpp
    spack load benchmark
fi
