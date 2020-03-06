# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(ConstitutiveRelations)

set(LIBRARY PotentialEnergy)

set(LIBRARY_SOURCES
  PotentialEnergy.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ConstitutiveRelations
  DataStructures
  Domain
  Elasticity
  Utilities
  )
