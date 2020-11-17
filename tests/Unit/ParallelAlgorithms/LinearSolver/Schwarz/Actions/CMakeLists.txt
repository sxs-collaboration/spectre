# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelSchwarzActions")

set(LIBRARY_SOURCES
  Test_CommunicateOverlapFields.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/Schwarz/Actions"
  "${LIBRARY_SOURCES}"
  "Convergence;DataStructures;DomainStructure;Informer;ParallelSchwarz;Spectral"
  )
