# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizonFinderCriteria)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  IncreaseResolution.cpp
  Residual.cpp
  Shape.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Criteria.hpp
  Criterion.hpp
  Factory.hpp
  IncreaseResolution.hpp
  Residual.hpp
  Shape.hpp
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  SphericalHarmonics
  Strahlkorper
  Parallel
  Serialization
  )

add_subdirectory(Tags)
