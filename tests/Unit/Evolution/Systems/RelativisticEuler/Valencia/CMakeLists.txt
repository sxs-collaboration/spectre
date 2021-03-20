# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Valencia")

set(LIBRARY_SOURCES
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  Test_Characteristics.cpp
  Test_ConservativeFromPrimitive.cpp
  Test_FixConservatives.cpp
  Test_Fluxes.cpp
  Test_PrimitiveFromConservative.cpp
  Test_Sources.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/RelativisticEuler/Valencia/"
  "${LIBRARY_SOURCES}"
  "Test_Domain;Test_GeneralRelativity;Test_Hydro;Valencia"
  )
