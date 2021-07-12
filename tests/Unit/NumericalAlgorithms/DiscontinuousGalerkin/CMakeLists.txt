# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(NumericalFluxes)
add_subdirectory(Tags)

set(LIBRARY "Test_NumericalDiscontinuousGalerkin")

set(LIBRARY_SOURCES
  Test_ApplyMassMatrix.cpp
  Test_Formulation.cpp
  Test_HasReceivedFromAllMortars.cpp
  Test_LiftFlux.cpp
  Test_MetricIdentityJacobian.cpp
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
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainStructure
  DiscontinuousGalerkin
  ErrorHandling
  Spectral
  Utilities
  )
