# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY PuncturesAmrCriteria)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  RefineAtPunctures.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  RefineAtPunctures.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Amr
  AmrCriteria
  DataStructures
  DomainStructure
  Elliptic
  InitialDataUtilities
  Options
  Parallel
  Serialization
  Utilities
  PRIVATE
  Domain
  PuncturesAnalyticData
  )
