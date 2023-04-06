# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Hydro)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ComovingMagneticField.cpp
  LorentzFactor.cpp
  MassFlux.cpp
  MassWeightedFluidItems.cpp
  SoundSpeedSquared.cpp
  SpecificEnthalpy.cpp
  StressEnergy.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComovingMagneticField.hpp
  LorentzFactor.hpp
  MassFlux.hpp
  MassWeightedFluidItems.hpp
  SoundSpeedSquared.hpp
  SpecificEnthalpy.hpp
  StressEnergy.hpp
  Tags.hpp
  TagsDeclarations.hpp
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
  )

add_subdirectory(EquationsOfState)
