# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GhGrMhdAnalyticData)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InstantiateWrappedGr.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  GeneralRelativitySolutions
  GrMhdAnalyticData
  PRIVATE
  DataStructures
  Utilities
  )
