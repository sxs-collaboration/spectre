# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Importers)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ObservationSelector.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ElementDataReader.hpp
  ObservationSelector.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ErrorHandling
  Options
  INTERFACE
  DataStructures
  DomainStructure
  IO
  Initialization
  Parallel
  Utilities
  )

add_subdirectory(Actions)
