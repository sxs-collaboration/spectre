# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ConstitutiveRelations)

set(LIBRARY_SOURCES
  ConstitutiveRelation.cpp
  IsotropicHomogeneous.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
