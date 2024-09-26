# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  KerrHorizon.cpp
  LineSegment.cpp
  SpecifiedPoints.cpp
  Sphere.cpp
  WedgeSectionTorus.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  KerrHorizon.hpp
  LineSegment.hpp
  SpecifiedPoints.hpp
  Sphere.hpp
  WedgeSectionTorus.hpp
)
