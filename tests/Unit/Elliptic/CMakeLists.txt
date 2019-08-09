# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)
add_subdirectory(DiscontinuousGalerkin)
add_subdirectory(Systems)

set(LIBRARY "Test_Elliptic")

set(LIBRARY_SOURCES
  Test_NumericInitialGuess.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic"
  "${LIBRARY_SOURCES}"
  "DataStructures"
  )
