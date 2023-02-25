# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Imex)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GuessResult.cpp
  Mode.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GuessResult.hpp
  Mode.hpp
  NamespaceDocs.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ErrorHandling
  Options
  INTERFACE
  DataStructures
  Time
  Utilities
  )

add_subdirectory(Protocols)
add_subdirectory(Tags)
