# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Valencia")

set(LIBRARY_SOURCES
  Test_Characteristics.cpp
  Test_ConservativeFromPrimitive.cpp
  Test_Equations.cpp
  Test_Fluxes.cpp
  Test_PrimitiveFromConservative.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/RelativisticEuler/Valencia/"
  "${LIBRARY_SOURCES}"
  "Valencia;Test_Domain"
  )
