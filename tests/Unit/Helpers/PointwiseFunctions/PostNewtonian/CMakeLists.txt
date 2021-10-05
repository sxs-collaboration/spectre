# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PostNewtonianHelpers")

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BinaryTrajectories.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BinaryTrajectories.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Utilities
  PUBLIC
  DataStructures
  )
