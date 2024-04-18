# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SphericalHarmonicsIO")

set(LIBRARY_SOURCES
  Test_FillYlmLegendAndData.cpp
  Test_ReadSurfaceYlm.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  H5
  ParallelInterpolation
  SphericalHarmonics
  SphericalHarmonicsIO
  Utilities
  )
