# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Punctures)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Sources.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FirstOrderSystem.hpp
  Punctures.hpp
  Sources.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Utilities
  INTERFACE
  Poisson
  )

add_subdirectory(BoundaryConditions)
