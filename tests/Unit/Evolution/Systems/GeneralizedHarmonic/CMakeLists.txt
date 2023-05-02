# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(GaugeSourceFunctions)
add_subdirectory(ConstraintDamping)

set(LIBRARY "Test_GeneralizedHarmonic")

set(LIBRARY_SOURCES
  Actions/Test_SetInitialData.cpp
  BoundaryConditions/Test_Bjorhus.cpp
  BoundaryConditions/Test_BjorhusImpl.cpp
  BoundaryConditions/Test_DemandOutgoingCharSpeeds.cpp
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_DirichletMinkowski.cpp
  BoundaryConditions/Test_Periodic.cpp
  BoundaryCorrections/Test_UpwindPenalty.cpp
  Test_Characteristics.cpp
  Test_Constraints.cpp
  Test_DuDt.cpp
  Test_DuDtTempTags.cpp
  Test_Fluxes.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GeneralizedHarmonic/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataBoxTestHelpers
  DataStructures
  DataStructuresHelpers
  Domain
  DomainStructure
  FunctionsOfTime
  GeneralRelativity
  GeneralRelativityHelpers
  GeneralRelativitySolutions
  GeneralizedHarmonic
  Importers
  LinearOperators
  MathFunctions
  Options
  Spectral
  Utilities
)
