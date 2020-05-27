# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(NumericalFluxes)

set(LIBRARY "Test_EllipticDG")

set(LIBRARY_SOURCES
  Test_ImposeBoundaryConditions.cpp
  Test_ImposeInhomogeneousBoundaryConditionsOnSource.cpp
  Test_InitializeFirstOrderOperator.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;ErrorHandling"
  )
