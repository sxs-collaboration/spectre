# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Burgers")

set(LIBRARY_SOURCES
  Test_Characteristics.cpp
  Test_Fluxes.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Burgers/"
  "${LIBRARY_SOURCES}"
  "Burgers"
  )
