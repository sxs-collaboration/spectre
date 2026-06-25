# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Detail.hpp
  Factory.hpp
  FilledCylinder.hpp
  FilledCylinder.tpp
  Filter.hpp
  HollowCylinder.hpp
  HollowCylinder.tpp
  Hypercube.hpp
  Hypercube.tpp
  None.hpp
  None.tpp
  SphericalShell.hpp
  SphericalShell.tpp
  Tag.hpp
  )
