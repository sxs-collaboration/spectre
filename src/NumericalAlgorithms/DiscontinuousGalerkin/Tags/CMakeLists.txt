# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  DiscontinuousGalerkin
  PRIVATE
  Formulation.cpp
  )

spectre_target_headers(
  DiscontinuousGalerkin
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Formulation.hpp
  OptionsGroup.hpp
  )
