# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Burgers")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  Test_Fluxes.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Burgers/"
  "${LIBRARY_SOURCES}"
  "Burgers"
  )
