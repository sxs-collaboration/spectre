# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEuler)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  ConservativeFromPrimitive.cpp
  Fluxes.cpp
  InternalEnergyDensity.cpp
  KineticEnergyDensity.cpp
  MachNumber.cpp
  PrimitiveFromConservative.cpp
  RamPressure.cpp
  SoundSpeedSquared.cpp
  SpecificKineticEnergy.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AllSolutions.hpp
  Characteristics.hpp
  ConservativeFromPrimitive.hpp
  Fluxes.hpp
  InternalEnergyDensity.hpp
  KineticEnergyDensity.hpp
  MachNumber.hpp
  PrimitiveFromConservative.hpp
  RamPressure.hpp
  SoundSpeedSquared.hpp
  SpecificKineticEnergy.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DgSubcell
  Domain
  ErrorHandling
  Evolution
  Hydro
  Options
  Serialization
  PRIVATE
  FiniteDifference
  NewtonianEulerAnalyticData
  NewtonianEulerSolutions
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Limiters)
add_subdirectory(Sources)
add_subdirectory(Subcell)
