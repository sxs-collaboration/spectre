# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Importers)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ElementDataReader.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  DomainStructure
  ErrorHandling
  IO
  Initialization
  Options
  Parallel
  Utilities
  )

add_subdirectory(Actions)
