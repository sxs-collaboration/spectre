# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_MonteCarloSolutions")

set(LIBRARY_SOURCES
  Test_HomogeneousSphere.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  GeneralRelativity
  GeneralRelativityHelpers
  GeneralRelativitySolutions
  H5
  Hydro
  MonteCarlo
  MonteCarloSolutions
  Utilities
  )
