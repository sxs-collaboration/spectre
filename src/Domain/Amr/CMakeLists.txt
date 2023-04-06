# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Amr)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Flag.cpp
  Helpers.cpp
  NeighborsOfChild.cpp
  NewNeighborIds.cpp
  UpdateAmrDecision.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Amr.hpp
  Flag.hpp
  Helpers.hpp
  NeighborsOfChild.hpp
  NewNeighborIds.hpp
  UpdateAmrDecision.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Domain
  DomainStructure
  ErrorHandling
  Utilities
  )

add_subdirectory(Tags)
