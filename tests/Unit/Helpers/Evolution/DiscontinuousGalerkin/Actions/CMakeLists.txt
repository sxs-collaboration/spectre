## Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "EvolutionDgActionsHelpers")

# We split the instantiations across multiple files to allow better
# parallelization of the build.
set(LIBRARY_SOURCES
  InstantiateBcOld1d.cpp
  InstantiateBcOld2d.cpp
  InstantiateBcOld3d.cpp
  SystemType.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ErrorHandling
  Evolution
  DiscontinuousGalerkin
  Domain
  DomainStructure
  Spectral
  Time
  Utilities
  )
