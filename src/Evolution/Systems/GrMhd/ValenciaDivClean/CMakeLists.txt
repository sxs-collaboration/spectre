# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ValenciaDivClean)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  ComovingMagneticFieldMagnitude.cpp
  ConservativeFromPrimitive.cpp
  FixConservatives.cpp
  Flattener.cpp
  Fluxes.cpp
  PrimitiveFromConservative.cpp
  PrimitiveFromConservativeOptions.cpp
  SetVariablesNeededFixingToFalse.cpp
  Sources.cpp
  TimeDerivativeTerms.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AllSolutions.hpp
  Characteristics.hpp
  ComovingMagneticFieldMagnitude.hpp
  ConservativeFromPrimitive.hpp
  FixConservatives.hpp
  Flattener.hpp
  Fluxes.hpp
  KastaunEtAl.hpp
  KastaunEtAl.tpp
  KastaunEtAlHydro.hpp
  KastaunEtAlHydro.tpp
  NewmanHamlin.hpp
  NewmanHamlin.tpp
  PalenzuelaEtAl.hpp
  PalenzuelaEtAl.tpp
  PrimitiveFromConservative.hpp
  PrimitiveFromConservativeOptions.hpp
  PrimitiveRecoveryData.hpp
  QuadrupoleFormula.hpp
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
  Evolution
  FiniteDifference
  GeneralRelativity
  GrMhdAnalyticData
  GrMhdSolutions
  H5
  Hydro
  HydroHelpers
  Limiters
  Options
  RelativisticEulerSolutions
  Utilities
  Valencia
  VariableFixing
  PRIVATE
  RootFinding
  Simd
  )

add_subdirectory(Actions)
add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Subcell)
