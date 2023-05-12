#
# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ComputeFluxes.hpp
  GhostData.cpp
  InitialDataTci.cpp
  SwapGrTags.cpp
  TciOnDgGrid.cpp
  TciOnFdGrid.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GhostData.hpp
  InitialDataTci.hpp
  Subcell.hpp
  SwapGrTags.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  )
