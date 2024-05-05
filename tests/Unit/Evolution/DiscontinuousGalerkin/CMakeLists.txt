# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Limiters)
add_subdirectory(Messages)

set(LIBRARY "Test_EvolutionDg")

set(LIBRARY_SOURCES
  Actions/Test_ApplyBoundaryCorrections.cpp
  Actions/Test_BoundaryConditions.cpp
  Actions/Test_ComputeTimeDerivative.cpp
  Actions/Test_NormalCovectorAndMagnitude.cpp
  Initialization/Test_Mortars.cpp
  Initialization/Test_QuadratureTag.cpp
  Test_BackgroundGrVars.cpp
  Test_BoundaryCorrectionsHelper.cpp
  Test_BoundaryData.cpp
  Test_MortarData.cpp
  Test_MortarTags.cpp
  Test_NormalVectorTags.cpp
  Test_UsingSubcell.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  CoordinateMaps
  DataStructures
  DataStructuresHelpers
  DiscontinuousGalerkin
  Domain
  DomainCreators
  DomainStructure
  DomainTimeDependence
  Evolution
  EvolutionDgActionsHelpers
  GeneralRelativitySolutions
  Hydro
  Options
  RelativisticEulerSolutions
  Spectral
  Time
  Utilities
  )
