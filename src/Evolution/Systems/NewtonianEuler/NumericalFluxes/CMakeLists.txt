# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY NewtonianEulerNumericalFluxes)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Hllc.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Hllc.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Options
  PRIVATE
  ErrorHandling
  )
