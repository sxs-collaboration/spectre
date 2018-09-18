# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ApparentHorizons)

set(LIBRARY_SOURCES
    FastFlow.cpp
    SpherepackIterator.cpp
    Strahlkorper.cpp
    StrahlkorperGr.cpp
    Tags.cpp
    YlmSpherepack.cpp
    YlmSpherepackHelper.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE GeneralRelativity
  INTERFACE LinearAlgebra
  INTERFACE SPHEREPACK
  )
