# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SphericalHarmonics")

set(LIBRARY_SOURCES
  Test_ChangeCenterOfStrahlkorper.cpp
  Test_RealSphericalHarmonics.cpp
  Test_Spherepack.cpp
  Test_SpherepackIterator.cpp
  Test_Strahlkorper.cpp
  Test_StrahlkorperFunctions.cpp
  Test_Tags.cpp
  Test_YlmToStf.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/SphericalHarmonics/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  SphericalHarmonics
  SphericalHarmonicsHelpers
  Utilities
  )
