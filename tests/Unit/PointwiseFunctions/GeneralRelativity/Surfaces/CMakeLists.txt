# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GrSurfaces")

set(LIBRARY_SOURCES
  Test_ComputeItems.cpp
  Test_GrSurfaces.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ApparentHorizonFinder
  DataStructures
  GeneralRelativity
  GeneralRelativitySolutions
  GrSurfaces
  GrSurfacesHelpers
  SphericalHarmonics
  SphericalHarmonicsHelpers
  Utilities
)
