# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainCreators")

set(LIBRARY_SOURCES
  Test_AlignedLattice.cpp
  Test_Brick.cpp
  Test_Cylinder.cpp
  Test_Disk.cpp
  Test_FrustalCloak.cpp
  Test_Interval.cpp
  Test_Rectangle.cpp
  Test_RotatedBricks.cpp
  Test_RotatedIntervals.cpp
  Test_RotatedRectangles.cpp
  Test_Shell.cpp
  Test_Sphere.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/Creators"
  "${LIBRARY_SOURCES}"
  "DomainCreators;Test_Domain"
  )
