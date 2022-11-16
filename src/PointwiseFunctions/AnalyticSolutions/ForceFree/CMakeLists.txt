# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ForceFreeSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AlfvenWave.cpp
  FastWave.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AlfvenWave.hpp
  Factory.hpp
  FastWave.hpp
  Solutions.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  Serialization
  )
