# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_XctsPointwiseFunctions)

set(LIBRARY_SOURCES
  Test_LongitudinalOperator.cpp
  Test_SpacetimeQuantities.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/Xcts/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Spectral
  Utilities
  XctsPointwiseFunctions
  )
