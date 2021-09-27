# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ApparentHorizon.cpp
  KerrHorizon.cpp
  LineSegment.cpp
  SpecifiedPoints.cpp
  WedgeSectionTorus.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApparentHorizon.hpp
  KerrHorizon.hpp
  LineSegment.hpp
  SpecifiedPoints.hpp
  WedgeSectionTorus.hpp
)
