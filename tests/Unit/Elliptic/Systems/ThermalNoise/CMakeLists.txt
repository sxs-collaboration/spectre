# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ThermalNoise")

set(LIBRARY_SOURCES
  Test_ThermalNoiseTags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/ThermalNoise/"
  "${LIBRARY_SOURCES}"
  ""
  )
