# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SphericalHarmonics")

set(LIBRARY_SOURCES
  Test_SpherepackIterator.cpp
  Test_YlmSpherepack.cpp
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
