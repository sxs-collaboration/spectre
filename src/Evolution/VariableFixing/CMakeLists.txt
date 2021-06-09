# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY VariableFixing)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FixToAtmosphere.cpp
  LimitLorentzFactor.cpp
  RadiallyFallingFloor.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Actions.hpp
  FixToAtmosphere.hpp
  LimitLorentzFactor.hpp
  RadiallyFallingFloor.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  GeneralRelativity
  Hydro
  Options
  Utilities
  )
