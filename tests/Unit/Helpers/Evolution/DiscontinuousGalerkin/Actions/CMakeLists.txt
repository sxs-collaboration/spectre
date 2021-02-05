## Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "EvolutionDgActionsHelpers")

# We split the instantiations across multiple files to allow better
# parallelization of the build.
set(LIBRARY_SOURCES
  InstantiateBcOld1d.cpp
  InstantiateBcOld2d.cpp
  InstantiateBcOld3d.cpp
  InstantiateCons1d.cpp
  InstantiateCons2d.cpp
  InstantiateCons3d.cpp
  InstantiateMixed1d.cpp
  InstantiateMixed2d.cpp
  InstantiateMixed3d.cpp
  InstantiateNoncons1d.cpp
  InstantiateNoncons2d.cpp
  InstantiateNoncons3d.cpp
  SystemType.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ErrorHandling
  Evolution
  Framework
  DiscontinuousGalerkin
  Domain
  DomainStructure
  Interpolation
  Spectral
  Time
  Utilities
  )
