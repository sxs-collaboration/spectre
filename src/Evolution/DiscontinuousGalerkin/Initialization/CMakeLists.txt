# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  Evolution
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Mortars.hpp
  QuadratureTag.hpp
  )

spectre_target_sources(
  Evolution
  PRIVATE
  Mortars.cpp
  )
