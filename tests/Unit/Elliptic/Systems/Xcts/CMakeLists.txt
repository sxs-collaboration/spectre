# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Xcts")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  Test_HydroQuantities.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Systems/Xcts/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  Domain
  Elliptic
  Hydro
  InitialDataUtilities
  Options
  Parallel
  Utilities
  Xcts
  XctsAnalyticData
  XctsSolutions
  )

add_subdirectory(BoundaryConditions)
