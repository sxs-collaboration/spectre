# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FixConservativesAndComputePrims.cpp
  InitialDataTci.cpp
  PrimsAfterRollback.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FixConservativesAndComputePrims.hpp
  InitialDataTci.hpp
  PrimsAfterRollback.hpp
  Subcell.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  )
