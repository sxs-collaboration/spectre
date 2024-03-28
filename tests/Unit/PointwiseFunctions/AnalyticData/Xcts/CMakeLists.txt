# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_XctsAnalyticData")

set(LIBRARY_SOURCES
  Test_Binary.cpp
  Test_BinaryWithGravitationalWaves.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  LinearOperators
  Spectral
  Utilities
  Xcts
  XctsAnalyticData
  XctsSolutions
  )
