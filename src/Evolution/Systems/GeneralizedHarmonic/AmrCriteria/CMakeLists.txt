# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralizedHarmonicAmrCriteria)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  SpecPRefinement.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  SpecPRefinement.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Amr
  AmrCriteria
  DataStructures
  Domain
  DomainStructure
  GeneralizedHarmonic
  GeneralRelativity
  LinearOperators
  Options
  Parallel
  Serialization
  Utilities
  PRIVATE
  ErrorHandling
  Spectral
  )
