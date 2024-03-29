# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ElasticityActions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InitializeConstitutiveRelation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitializeConstitutiveRelation.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ConstitutiveRelations
  DomainCreators
  INTERFACE
  DataStructures
  Domain
  Parallel
  Utilities
  )
