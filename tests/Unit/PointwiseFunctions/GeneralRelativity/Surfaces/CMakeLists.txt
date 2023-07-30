# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_StrahlkorperGr")

set(LIBRARY_SOURCES
  Test_StrahlkorperGr.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/GeneralRelativity/Surfaces"
  "${LIBRARY_SOURCES}"
  ""
  )

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
  Utilities
)
