# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  TciOnDgGrid.cpp
  TciOnFdGrid.cpp
  TciOptions.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Subcell.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  TciOptions.hpp
  )
