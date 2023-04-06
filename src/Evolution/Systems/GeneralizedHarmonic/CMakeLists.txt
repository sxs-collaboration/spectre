# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralizedHarmonic)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Constraints.cpp
  Equations.cpp
  TimeDerivative.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AllSolutions.hpp
  Characteristics.hpp
  Constraints.hpp
  DuDtTempTags.hpp
  Equations.hpp
  Initialize.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivative.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  CoordinateMaps
  DataStructures
  Domain
  DomainBoundaryConditions
  ErrorHandling
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
  Events # For observer tags
  Initialization
  Parallel
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(ConstraintDamping)
add_subdirectory(GaugeSourceFunctions)
