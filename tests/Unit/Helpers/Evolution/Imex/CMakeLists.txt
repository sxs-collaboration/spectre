# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "ImexHelpers")

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/Unit
  HEADERS
  DoImplicitStepSector.hpp
  TestSector.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DoImplicitStepInstantiate.cpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Imex
  Utilities
  )
