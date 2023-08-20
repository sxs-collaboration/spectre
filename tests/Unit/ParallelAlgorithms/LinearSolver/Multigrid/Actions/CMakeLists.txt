# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelMultigridActions")

set(LIBRARY_SOURCES
  Test_RestrictFields.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Convergence
  DataStructures
  Domain
  DomainStructure
  Logging
  Parallel
  ParallelMultigrid
  Spectral
  Utilities
  )
