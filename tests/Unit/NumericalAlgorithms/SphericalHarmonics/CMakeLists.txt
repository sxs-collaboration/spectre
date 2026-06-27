# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SphericalHarmonics")

set(LIBRARY_SOURCES
  Test_AngularOrdering.cpp
  Test_RealSphericalHarmonics.cpp
  Test_Spherepack.cpp
  Test_SpherepackIterator.cpp
  Test_YlmToStf.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  SphericalHarmonics
  SphericalHarmonicsHelpers
  SpinWeightedSphericalHarmonics
  Utilities
  )

add_subdirectory(Python)
