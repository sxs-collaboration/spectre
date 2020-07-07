# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY M1Grey)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Fluxes.cpp
  M1Closure.cpp
  M1HydroCoupling.cpp
  Sources.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Fluxes.hpp
  Initialize.hpp
  M1Closure.hpp
  M1HydroCoupling.hpp
  Sources.hpp
  System.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE GeneralRelativity
  INTERFACE Hydro
  INTERFACE RootFinding
  INTERFACE Utilities
  )
