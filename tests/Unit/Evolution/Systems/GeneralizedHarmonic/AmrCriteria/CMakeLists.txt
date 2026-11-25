# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GeneralizedHarmonicAmrCriteria")

set(LIBRARY_SOURCES
  Test_SpecPRefinement.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Amr
  AmrCriteria
  DataStructures
  Domain
  DomainStructure
  GeneralizedHarmonic
  GeneralizedHarmonicAmrCriteria
  GeneralRelativity
  LinearOperators
  Options
  Parallel
  Spectral
  Utilities
  )
