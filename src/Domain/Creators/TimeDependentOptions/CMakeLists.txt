# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BinaryCompactObject.cpp
  FromVolumeFile.cpp
  ShapeMap.cpp
  Sphere.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BinaryCompactObject.hpp
  FromVolumeFile.hpp
  ShapeMap.hpp
  Sphere.hpp
  )
