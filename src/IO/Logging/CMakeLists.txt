# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Logging)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Verbosity.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Tags.hpp
  Verbosity.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Options
  PRIVATE
  ErrorHandling
  INTERFACE
  DataStructures
  )
