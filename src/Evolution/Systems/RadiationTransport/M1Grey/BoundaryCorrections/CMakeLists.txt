# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  M1Grey
  PRIVATE
  Rusanov.cpp
  )

spectre_target_headers(
  M1Grey
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCorrection.hpp
  Rusanov.hpp
  )
