# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEuler")

set(LIBRARY_SOURCES
  Test_Characteristics.cpp
  Test_ComputeItems.cpp
  Test_ConservativeFromPrimitive.cpp
  Test_Fluxes.cpp
  Test_PrimitiveFromConservative.cpp
  Test_SoundSpeedSquared.cpp
  Test_Sources.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_subdirectory(Limiters)
add_subdirectory(NumericalFluxes)
add_subdirectory(Sources)

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/NewtonianEuler/"
  "${LIBRARY_SOURCES}"
  "NewtonianEuler;Test_Domain"
  )
