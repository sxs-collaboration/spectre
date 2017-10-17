\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Installation on Clusters {#installation_on_clusters}

## Wheeler at Caltech

1. Clone SpECTRE into `$SPECTRE_HOME`
2. Run
   `mkdir $SPECTRE_HOME/build_[gcc|clang] && cd $SPECTRE_HOME/build_[gcc|clang]`
3. Run `. $SPECTRE_HOME/support/Environments/wheeler_[gcc|llvm].env` to load
   the GCC or LLVM/Clang environment
4. Run `cmake -D CMAKE_BUILD_TYPE=[Release|Debug]
   -D CMAKE_Fortran_COMPILER=gfortran $SPECTRE_HOME`
5. Run `make -j4`
