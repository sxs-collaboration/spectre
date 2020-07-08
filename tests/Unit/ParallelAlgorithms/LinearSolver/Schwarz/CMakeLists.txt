# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelSchwarz")

set(LIBRARY_SOURCES
  Test_ElementCenteredSubdomainData.cpp
  Test_OverlapHelpers.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/Schwarz"
  "${LIBRARY_SOURCES}"
  "DataStructures;DomainStructure;LinearSolver;ParallelSchwarz"
  )
