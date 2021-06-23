# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FixConservativesAndComputePrims.cpp
  InitialDataTci.cpp
  PrimitiveGhostData.cpp
  PrimsAfterRollback.cpp
  ResizeAndComputePrimitives.cpp
  SwapGrTags.cpp
  TciOnDgGrid.cpp
  TciOnFdGrid.cpp
  TciOptions.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComputeFluxes.hpp
  FixConservativesAndComputePrims.hpp
  GrTagsForHydro.hpp
  InitialDataTci.hpp
  PrimitiveGhostData.hpp
  PrimsAfterRollback.hpp
  ResizeAndComputePrimitives.hpp
  Subcell.hpp
  SwapGrTags.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  TciOptions.hpp
  TimeDerivative.hpp
  )
