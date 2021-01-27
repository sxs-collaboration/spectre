# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Burgers")

set(LIBRARY_SOURCES
  BoundaryCorrections/Test_Hll.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  Test_Characteristics.cpp
  Test_Fluxes.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Burgers/"
  "${LIBRARY_SOURCES}"
  "Burgers"
  )
