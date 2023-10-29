# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Tags)

set(LIBRARY "Test_NumericalDiscontinuousGalerkin")

set(LIBRARY_SOURCES
  Test_ApplyMassMatrix.cpp
  Test_Formulation.cpp
  Test_HasReceivedFromAllMortars.cpp
  Test_InterpolateFromBoundary.cpp
  Test_LiftFlux.cpp
  Test_LiftFromBoundary.cpp
  Test_MetricIdentityJacobian.cpp
  Test_MortarHelpers.cpp
  Test_NormalDotFlux.cpp
  Test_ProjectToBoundary.cpp
  Test_SimpleBoundaryData.cpp
  Test_SimpleMortarData.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainStructure
  DiscontinuousGalerkin
  ErrorHandling
  Interpolation
  Spectral
  Utilities
  )
