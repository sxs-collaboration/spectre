# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_StrahlkorperGr")

set(LIBRARY_SOURCES
  Test_ComputeItems.cpp
  Test_StrahlkorperGr.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ApparentHorizons
  DataStructures
  GeneralRelativity
  GeneralRelativitySolutions
  StrahlkorperGr
  StrahlkorperGrHelpers
  SphericalHarmonics
  SphericalHarmonicsHelpers
  Utilities
)
