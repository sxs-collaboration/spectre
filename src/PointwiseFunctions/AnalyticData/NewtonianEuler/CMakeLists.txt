# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEulerAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  KhInstability.cpp
  ShuOsherTube.cpp
  SodExplosion.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  KhInstability.hpp
  ShuOsherTube.hpp
  SodExplosion.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC DataStructures
  INTERFACE ErrorHandling
  INTERFACE Hydro
  INTERFACE NewtonianEuler
  )
