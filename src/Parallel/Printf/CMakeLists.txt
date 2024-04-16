# Distributed under the MIT License.
# See LICENSE.txt for details.

add_charm_module(Printf)

set(LIBRARY Printf)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Printf.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Printf.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ErrorHandling
  INTERFACE
  Utilities
  PRIVATE
  SystemUtilities
  )

add_dependencies(
  ${LIBRARY}
  module_Printf
  )
