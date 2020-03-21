# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CoordinateMaps)

set(LIBRARY_SOURCES
    Affine.cpp
    BulgedCube.cpp
    CubicScale.cpp
    DiscreteRotation.cpp
    EquatorialCompression.cpp
    Equiangular.cpp
    Frustum.cpp
    Identity.cpp
    Rotation.cpp
    RotationTimeDep.cpp
    SpecialMobius.cpp
    Translation.cpp
    Wedge2D.cpp
    Wedge3D.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  ErrorHandling
  FunctionsOfTime

  INTERFACE
  Domain  # Domain should really be PUBLIC but there are cyclic dependencies
  )
