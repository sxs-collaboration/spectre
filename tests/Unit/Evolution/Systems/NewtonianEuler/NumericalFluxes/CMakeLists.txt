# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_NewtonianEulerNumericalFluxes")

set(LIBRARY_SOURCES
  Test_Hllc.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/NewtonianEuler/NumericalFluxes"
  "${LIBRARY_SOURCES}"
  "NewtonianEulerNumericalFluxes"
  )
