# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativity)

set(LIBRARY_SOURCES
    Christoffel.cpp
    ComputeGhQuantities.cpp
    ComputeSpacetimeQuantities.cpp
    IndexManipulation.cpp
    KerrSchildCoords.cpp
    Ricci.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
