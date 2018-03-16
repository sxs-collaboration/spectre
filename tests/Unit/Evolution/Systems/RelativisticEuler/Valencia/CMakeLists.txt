# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Valencia")

set(LIBRARY_SOURCES
  Test_ConservativeFromPrimitive.cpp
  Test_Fluxes.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/RelativisticEuler/Valencia/"
  "${LIBRARY_SOURCES}"
  "Valencia"
  )
