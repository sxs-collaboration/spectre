# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ValenciaDivClean")

set(LIBRARY_SOURCES
  Test_Characteristics.cpp
  Test_ConservativeFromPrimitive.cpp
  Test_FixConservatives.cpp
  Test_Fluxes.cpp
  Test_PrimitiveFromConservative.cpp
  Test_Sources.cpp
  Test_ValenciaDivClean.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GrMhd/ValenciaDivClean"
  "${LIBRARY_SOURCES}"
  "ValenciaDivClean"
  )
