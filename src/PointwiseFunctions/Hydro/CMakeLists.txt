# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Hydro)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ComovingMagneticField.cpp
  InversePlasmaBeta.cpp
  LorentzFactor.cpp
  LowerSpatialFourVelocity.cpp
  MagneticFieldTreatment.cpp
  MassFlux.cpp
  MassWeightedFluidItems.cpp
  QuadrupoleFormula.cpp
  SoundSpeedSquared.cpp
  SpecificEnthalpy.cpp
  StressEnergy.cpp
  TransportVelocity.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComovingMagneticField.hpp
  InversePlasmaBeta.hpp
  LorentzFactor.hpp
  LowerSpatialFourVelocity.hpp
  MagneticFieldTreatment.hpp
  MassFlux.hpp
  MassWeightedFluidItems.hpp
  QuadrupoleFormula.hpp
  SoundSpeedSquared.hpp
  SpecificEnthalpy.hpp
  StressEnergy.hpp
  Tags.hpp
  TagsDeclarations.hpp
  Temperature.hpp
  TransportVelocity.hpp
  Units.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  H5
  Options
  Serialization
  INTERFACE
  GeneralRelativity
  )

add_subdirectory(EquationsOfState)
add_subdirectory(InitialData)
add_subdirectory(Python)
