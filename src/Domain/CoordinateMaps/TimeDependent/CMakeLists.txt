# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CubicScale.cpp
  Rotation.cpp
  SphericalCompression.cpp
  Translation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CubicScale.hpp
  ProductMaps.hpp
  ProductMaps.tpp
  Rotation.hpp
  SphericalCompression.hpp
  Translation.hpp
  )
