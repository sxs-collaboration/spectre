# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY IO)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ComposeTable.cpp
  Connectivity.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComposeTable.hpp
  Connectivity.hpp
  )

# Notes:
# - The H5 lib depends in IO for the connectivity. Keep that in mind when adding
#   dependencies here.
target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Utilities
  )

add_subdirectory(Exporter)
add_subdirectory(External)
add_subdirectory(H5)
add_subdirectory(Logging)

# These two libraries should move to ParallelAlgorithms
add_subdirectory(Importers)
add_subdirectory(Observer)
