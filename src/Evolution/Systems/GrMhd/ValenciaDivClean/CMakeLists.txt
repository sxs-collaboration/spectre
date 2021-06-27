# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ValenciaDivClean)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  ConservativeFromPrimitive.cpp
  FixConservatives.cpp
  Fluxes.cpp
  KastaunEtAl.cpp
  NewmanHamlin.cpp
  PalenzuelaEtAl.cpp
  PrimitiveFromConservative.cpp
  SetVariablesNeededFixingToFalse.cpp
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
  KastaunEtAl.hpp
  NewmanHamlin.hpp
  PalenzuelaEtAl.hpp
  PrimitiveFromConservative.hpp
  PrimitiveRecoveryData.hpp
  SetVariablesNeededFixingToFalse.hpp
  Sources.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Domain
  DgSubcell
  DiscontinuousGalerkin
  ErrorHandling
  GeneralRelativity
  Hydro
  Limiters
  Options
  Utilities
  Valencia
  VariableFixing
  PRIVATE
  RootFinding
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Subcell)
