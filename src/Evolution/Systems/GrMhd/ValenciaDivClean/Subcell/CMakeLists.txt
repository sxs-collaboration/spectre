# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InitialDataTci.cpp
  SwapGrTags.cpp
  TciOnDgGrid.cpp
  TciOnFdGrid.cpp
  TciOptions.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitialDataTci.hpp
  Subcell.hpp
  SwapGrTags.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  TciOptions.hpp
  )
