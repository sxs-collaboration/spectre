# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DataImporter")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  Test_VolumeDataReaderActions.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

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

function(add_algorithm_test TEST_NAME DIM)
  set(TEST_BASE ${TEST_NAME}${DIM}D)
  set(CPP_NAME Test_${TEST_NAME})
  set(EXECUTABLE_NAME ${CPP_NAME}${DIM}D)
  set(TEST_IDENTIFIER Integration.Importers.${TEST_BASE})

  add_spectre_executable(
    ${EXECUTABLE_NAME}
    EXCLUDE_FROM_ALL
    ${CPP_NAME}.cpp
    )
  target_compile_definitions(
    ${EXECUTABLE_NAME}
    PRIVATE
    DIM=${DIM}
    )
  target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
    Charmxx::main
    Catch2::Catch2
    DataStructures
    DgSubcell
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
