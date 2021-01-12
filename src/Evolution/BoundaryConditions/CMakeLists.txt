# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EvolutionBoundaryConditions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Type.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Type.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ErrorHandling
  )
