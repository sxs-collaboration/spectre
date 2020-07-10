# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Hydro)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  LorentzFactor.cpp
  MassFlux.cpp
  SoundSpeedSquared.cpp
  SpecificEnthalpy.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  LorentzFactor.hpp
  MassFlux.hpp
  SoundSpeedSquared.hpp
  SpecificEnthalpy.hpp
  Tags.hpp
  TagsDeclarations.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  Options
  )

add_subdirectory(EquationsOfState)
