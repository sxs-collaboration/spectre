# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ElasticityPointwiseFunctions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  PotentialEnergy.cpp
  Strain.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  PotentialEnergy.hpp
  Strain.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ConstitutiveRelations
  DataStructures
  Domain
  Elasticity
  Utilities
  PRIVATE
  LinearOperators
  )

add_subdirectory(ConstitutiveRelations)
