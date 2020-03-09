# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)
add_subdirectory(BoundarySchemes)
add_subdirectory(NumericalFluxes)

set(LIBRARY "Test_NumericalDiscontinuousGalerkin")

set(LIBRARY_SOURCES
  Test_Formulation.cpp
  Test_LiftFlux.cpp
  Test_MortarHelpers.cpp
  Test_NormalDotFlux.cpp
  Test_Protocols.cpp
  Test_SimpleBoundaryData.cpp
  Test_SimpleMortarData.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "NumericalAlgorithms/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;DiscontinuousGalerkin;ErrorHandling;Spectral;Utilities"
  )
