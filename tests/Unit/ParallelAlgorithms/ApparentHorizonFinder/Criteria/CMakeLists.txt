# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ApparentHorizonFinderCriteria")

set(LIBRARY_SOURCES
  Test_Criterion.cpp
  Test_IncreaseResolution.cpp
  Test_Residual.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ApparentHorizonFinder
  ApparentHorizonFinderCriteria
  Options
  SphericalHarmonics
  )
