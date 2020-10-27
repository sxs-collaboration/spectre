# Distributed under the MIT License.
# See LICENSE.txt for details.

# This executable builds only core components for testing Blaze vectorization
# on different architectures with different compilers.
#
# It can be used to test that vectorization with Blaze correctly works on all
# of our target architectures without needing access to the CPUs. This is
# made possible by the Intel Software Development Emulator.
#
# The relevant architectures as of 2020 and their Intel SDE flags are:
# SDE Flag  march Flag                                Architecture
# -nhm      -nehalem                                  Nehalem CPU
# -wsm      -westmere                                 Westmere CPU
# -snb      -sandybridge                              Sandy Bridge CPU
# -ivb      -ivybridge                                Ivy Bridge CPU
# -hsw      -haswell                                  Haswell CPU
# -bdw      -broadwell                                Broadwell CPU
# -skl      -skylake                                  Skylake CPU
# -clx      -cascadelake                              Cascade Lake CPU
# -cpx      -cooperlake                               Cooper Lake CPU
# -icx      -icelake-server                           Ice Lake server CPU
# -tgl      -tigerlake                                Tiger Lake CPU
# -adl      -alderlake                                Alder Lake CPU
# -spr      -sapphirerapids                           Sapphire Rapids CPU
#
# Note that KNL is not listed. KNL-type chips have been discontinued by Intel
# due to poor performance.
set(EXECUTABLE "TestArchitectureVectorization")

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  ${EXECUTABLE}.cpp

  # The following lines list the source files with tests in them to be
  # compiled into the executable.
  ../DataStructures/Test_DataVector.cpp
  ../DataStructures/Test_DataVectorBinaryOperations.cpp
  ../DataStructures/Test_DenseMatrix.cpp
  ../DataStructures/Test_ComplexDataVector.cpp
  ../DataStructures/Test_ComplexDataVectorBinaryOperations.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Boost::program_options
  DataStructures
  ErrorHandling
  Framework
  Informer
  Parallel
  SystemUtilities
  )

add_dependencies(
  ${EXECUTABLE}
  module_GlobalCache
  module_Main
  module_RunTests
  )
