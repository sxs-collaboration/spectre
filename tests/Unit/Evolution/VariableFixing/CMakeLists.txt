# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_VariableFixing")

set(LIBRARY_SOURCES
  Test_Actions.cpp
  Test_FixToAtmosphere.cpp
  Test_LimitLorentzFactor.cpp
  Test_ParameterizedDeleptonization.cpp
  Test_RadiallyFallingFloor.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  Hydro
  Parallel
  VariableFixing
  Utilities
  )
