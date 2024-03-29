# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Valencia)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  ConservativeFromPrimitive.cpp
  FixConservatives.cpp
  Fluxes.cpp
  PrimitiveFromConservative.cpp
  Sources.cpp
  TimeDerivativeTerms.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  ConservativeFromPrimitive.hpp
  FixConservatives.hpp
  Fluxes.hpp
  PrimitiveFromConservative.hpp
  Sources.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  ErrorHandling
  Evolution
  GeneralRelativity
  Hydro
  Limiters
  LinearOperators
  Options
  Serialization
  Utilities
  VariableFixing
  PRIVATE
  RootFinding
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
