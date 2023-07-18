# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelSchwarz")

set(LIBRARY_SOURCES
  Test_ComputeTags.cpp
  Test_ElementCenteredSubdomainData.cpp
  Test_OverlapHelpers.cpp
  Test_Tags.cpp
  Test_Weighting.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/LinearSolver/Schwarz"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainStructure
  LinearSolver
  ParallelSchwarz
  Spectral
  )

add_standalone_test(
  "Integration.LinearSolver.SchwarzAlgorithm"
  INPUT_FILE "Test_SchwarzAlgorithm.yaml")
target_link_libraries(
  "Test_SchwarzAlgorithm"
  PRIVATE
  "${DISTRIBUTED_INTEGRATION_TEST_LINK_LIBRARIES};Domain;DomainStructure;ParallelSchwarz")

add_subdirectory(Actions)
