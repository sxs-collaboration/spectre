# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_XctsAnalyticData")

set(LIBRARY_SOURCES
  Test_Binary.cpp
  Test_KerrSchildTeukolsky.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  GeneralRelativitySolutions
  Spectral
  Utilities
  Xcts
  XctsAnalyticData
  XctsSolutions
  )
