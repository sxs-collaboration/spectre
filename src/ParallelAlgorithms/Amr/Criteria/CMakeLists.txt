# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY AmrCriteria)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DriveToTarget.cpp
  Random.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Criteria.hpp
  Criterion.hpp
  DriveToTarget.hpp
  Random.hpp
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Amr
  DomainStructure
  Options
  Parallel
  Spectral
  Utilities
  )

add_subdirectory(Tags)
