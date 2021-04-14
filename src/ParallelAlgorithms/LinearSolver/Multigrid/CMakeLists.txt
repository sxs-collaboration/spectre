# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelMultigrid)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Hierarchy.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Hierarchy.hpp
  Multigrid.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DomainStructure
  PRIVATE
  ErrorHandling
  Utilities
  )
