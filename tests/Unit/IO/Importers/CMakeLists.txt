# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DataImporter")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  Test_VolumeDataReaderActions.cpp
  )

add_test_library(
  ${LIBRARY}
  "IO/Importers"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainStructure
  IO
  Importers
  Options
  ParallelHelpers
  Spectral
  Utilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

function(add_algorithm_test TEST_NAME DIM)
  set(TEST_BASE ${TEST_NAME}${DIM}D)
  set(HPP_NAME Test_${TEST_NAME})
  set(EXECUTABLE_NAME ${HPP_NAME}${DIM}D)
  set(TEST_IDENTIFIER Integration.Importers.${TEST_BASE})

  add_spectre_parallel_executable(
    ${EXECUTABLE_NAME}
    ${HPP_NAME}
    IO/Importers
    Metavariables<${DIM}>
    ""
    )

  target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
    DataStructures
    Domain
    DomainCreators
    DomainStructure
    ErrorHandling
    Importers
    Informer
    IO
    Options
    Parallel
    SystemUtilities
    Utilities
    )

  add_standalone_test(
    "${TEST_IDENTIFIER}"
    INPUT_FILE "Test_${TEST_BASE}.yaml")
endfunction()

add_algorithm_test("VolumeDataReaderAlgorithm" 1)
add_algorithm_test("VolumeDataReaderAlgorithm" 2)
add_algorithm_test("VolumeDataReaderAlgorithm" 3)
