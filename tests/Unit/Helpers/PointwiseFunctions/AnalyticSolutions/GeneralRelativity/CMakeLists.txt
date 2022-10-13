# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "GrSolutionsTestHelpers")

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CheckWrappedGrConsistency.hpp
  VerifyGrSolution.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  CoordinateMaps
  DataStructures
  Domain
  GeneralizedHarmonic
  GeneralRelativity
  GeneralRelativitySolutions
  LinearOperators
  Spectral
  Utilities
  )
