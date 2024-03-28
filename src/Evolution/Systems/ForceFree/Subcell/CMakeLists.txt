#
# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ComputeFluxes.hpp
  GhostData.cpp
  NeighborPackagedData.cpp
  SetInitialRdmpData.cpp
  TciOnDgGrid.cpp
  TciOnFdGrid.cpp
  TciOptions.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  GhostData.hpp
  NeighborPackagedData.hpp
  SetInitialRdmpData.hpp
  Subcell.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  TciOptions.hpp
  )
