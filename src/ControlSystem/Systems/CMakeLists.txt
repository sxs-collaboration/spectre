# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Expansion.hpp
  GridCenters.hpp
  Rotation.hpp
  Shape.hpp
  Size.hpp
  Skew.hpp
  Translation.hpp
  )
