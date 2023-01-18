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
  H5
  Observer
  Options
  INTERFACE
  DataStructures
  DomainStructure
  Initialization
  Interpolation
  Parallel
  Utilities
  )

add_subdirectory(Actions)
