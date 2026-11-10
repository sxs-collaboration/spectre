# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianMhdAnalyticData")

set(LIBRARY_SOURCES
  Test_KhInstability.cpp
  Test_ShuOsherTube.cpp
  Test_SodExplosion.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  NewtonianMhdAnalyticData
  Options
  Utilities
  )
