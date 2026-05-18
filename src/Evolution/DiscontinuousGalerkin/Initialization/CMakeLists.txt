# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Mortars.hpp
  ProjectSpectralFilters.hpp
  QuadratureTag.hpp
  SetupEqualRateRegions.hpp
  SpectralFilters.hpp
  SpectralFilters.tpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Mortars.cpp
  SetupEqualRateRegions.cpp
  )
