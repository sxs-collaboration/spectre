# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralizedHarmonic)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ApplyTensorYlmFilter.cpp
  Characteristics.cpp
  Constraints.cpp
  Equations.cpp
  TimeDerivative.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyTensorYlmFilter.hpp
  Characteristics.hpp
  Constraints.hpp
  DuDtTempTags.hpp
  Equations.hpp
  Initialize.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivative.hpp
  TimeDerivative.tpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  CoordinateMaps
  ConstraintDamping
  DataStructures
  Domain
  DomainBoundaryConditions
  ErrorHandling
  Events
  Evolution
  FunctionsOfTime
  GeneralRelativity
  GeneralRelativitySolutions
  GhGrMhdAnalyticData
  GhGrMhdSolutions
  GhRelativisticEulerSolutions
  IO
  Importers
  Initialization
  LinearOperators
  Options
  Serialization
  Spectral
  Utilities
  INTERFACE
  Initialization
  Parallel
  )

add_subdirectory(Actions)
add_subdirectory(Bbh)
add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(GaugeSourceFunctions)
add_subdirectory(Instantiations)
