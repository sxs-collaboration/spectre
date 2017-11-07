# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CoordinateMaps)

set(LIBRARY_SOURCES
    Affine.cpp
    BulgedCube.cpp
    Equiangular.cpp
    Identity.cpp
    Rotation.cpp
    Wedge2D.cpp
    Wedge3D.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
