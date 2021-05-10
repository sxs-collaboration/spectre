# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "XctsSolutionsTestHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  VerifySolution.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  CoordinateMaps
  DataStructures
  Domain
  LinearOperators
  Spectral
  Utilities
  Xcts
  )
