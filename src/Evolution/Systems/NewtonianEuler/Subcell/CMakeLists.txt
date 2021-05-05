# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InitialDataTci.cpp
  PrimsAfterRollback.cpp
  ResizeAndComputePrimitives.cpp
  TciOnDgGrid.cpp
  TciOnFdGrid.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitialDataTci.hpp
  PrimsAfterRollback.hpp
  ResizeAndComputePrimitives.hpp
  Subcell.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  )
