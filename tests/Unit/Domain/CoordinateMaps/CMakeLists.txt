# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_COORDINATE_MAPS
  Domain/CoordinateMaps/Test_Affine.cpp
  Domain/CoordinateMaps/Test_BulgedCube.cpp
  Domain/CoordinateMaps/Test_CoordinateMap.cpp
  Domain/CoordinateMaps/Test_Equiangular.cpp
  Domain/CoordinateMaps/Test_Identity.cpp
  Domain/CoordinateMaps/Test_ProductMaps.cpp
  Domain/CoordinateMaps/Test_Rotation.cpp
  Domain/CoordinateMaps/Test_Wedge2D.cpp
  Domain/CoordinateMaps/Test_Wedge3D.cpp
  )

set(SPECTRE_UNIT_DOMAIN
  "${SPECTRE_UNIT_DOMAIN};${SPECTRE_UNIT_COORDINATE_MAPS}"
  PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};CoordinateMaps" PARENT_SCOPE)


