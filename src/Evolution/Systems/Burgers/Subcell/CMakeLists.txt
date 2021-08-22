# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GhostData.cpp
  InitialDataTci.cpp
  TciOnDgGrid.cpp
  TciOnFdGrid.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComputeFluxes.hpp
  GhostData.hpp
  InitialDataTci.hpp
  NeighborPackagedData.hpp
  Subcell.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  TciOptions.hpp
  TimeDerivative.hpp
  )
