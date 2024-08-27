# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarGaussBonnetPointwiseFunctions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ScalarMomentum.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ScalarMomentum.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  CurvedScalarWave
  DataStructures
  GeneralRelativity
  Utilities
  ScalarGaussBonnet
  Xcts
  )
