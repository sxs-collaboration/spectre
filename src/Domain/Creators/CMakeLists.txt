# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainCreators)

set(LIBRARY_SOURCES
  AlignedLattice.cpp
  Brick.cpp
  Cylinder.cpp
  Disk.cpp
  FrustalCloak.cpp
  Interval.cpp
  Rectangle.cpp
  RegisterDerivedWithCharm.cpp
  RotatedBricks.cpp
  RotatedIntervals.cpp
  RotatedRectangles.cpp
  Shell.cpp
  Sphere.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE CoordinateMaps
  INTERFACE DataStructures
  INTERFACE Domain
  INTERFACE ErrorHandling
  )
