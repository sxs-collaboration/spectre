# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Actions)

add_subdirectory(PhaseControl)

set(ALGORITHM_TEST_LINK_LIBRARIES
  # Link against Boost::program_options for now until we have proper
  # dependency handling for header-only libs
  Actions
  Boost::program_options
  Charmxx::main
  ErrorHandling
  Informer
  IO
  Options
  Parallel
  ParallelHelpers
  PhaseControl
  SystemUtilities
  Utilities
  )

# Add unit tests. For tests that result in a failure it is necessary to
# redirect output from stderr to stdout. However, it was necessary at least on
# some systems to do this redirect inside a shell command.
find_program(SHELL_EXECUTABLE "sh")
if (NOT SHELL_EXECUTABLE)
  message(FATAL_ERROR
    "Could not find 'sh' shell to execute failure tests of algorithms")
endif()

# Note that we have a 'balance' here that happens 2.5e4 of times per second just
# to be completely certain that the desired number of migrations occur during
# the test (2). This tends to have a lot of fluctuation depending on how quickly
# the machine manages to execute this fairly short test, and the current balance
# frequency has been roughly tuned to a) very consistently get at least 5
# balances (usually gets several dozen), and b) complete the test in a
# reasonable duration. Balancing too frequently can mean the test time explodes
# due to balancing so much that the test computation doesn't get much of a
# chance to execute.
function(add_algorithm_test_with_balancing TEST_NAME EXECUTABLE_NAME REGEX_TO_MATCH)
  add_standalone_test_executable("Test_${EXECUTABLE_NAME}")
  target_link_libraries(
    "Test_${EXECUTABLE_NAME}"
    PRIVATE
    "${ALGORITHM_TEST_LINK_LIBRARIES}")
  add_test(
    NAME "Unit.Parallel.${TEST_NAME}"
    COMMAND
    ${SHELL_EXECUTABLE}
    -c "${SPECTRE_TEST_RUNNER} ${CMAKE_BINARY_DIR}/bin/Test_${EXECUTABLE_NAME} \
    +p2 +balancer RotateLB +LBPeriod 4.0e-5 +LBDebug 3 2>&1"
    )

  set_standalone_test_properties("Unit.Parallel.${TEST_NAME}")
  if("${REGEX_TO_MATCH}" STREQUAL "")
    set_tests_properties(
      "Unit.Parallel.${TEST_NAME}"
      PROPERTIES
      FAIL_REGULAR_EXPRESSION "ERROR")
  else()
    set_tests_properties(
      "Unit.Parallel.${TEST_NAME}"
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "${REGEX_TO_MATCH}"
      FAIL_REGULAR_EXPRESSION "ERROR")
  endif()
endfunction()

# Run TEST_NAME twice: first with no command-line args, then with the
# `+restart SpectreCheckpoint0` command-line arg to restart from a Charm++
# checkpoint named `SpectreCheckpoint0`. The checkpoint file is then deleted
# to avoid it getting in the way of future test invocations (which would try
# to overwrite the prior checkpoint file and error).
function(add_algorithm_test_with_restart_from_checkpoint TEST_NAME)
  add_standalone_test_executable("Test_${TEST_NAME}")
  target_link_libraries(
    "Test_${TEST_NAME}"
    PRIVATE
    "${ALGORITHM_TEST_LINK_LIBRARIES}")
  # Executable writes checkpoint file into the dir where it runs:
  set(CHECKPOINT
    "${CMAKE_BINARY_DIR}/tests/Unit/Parallel/SpectreCheckpoint000000")
  add_test(
    NAME "Unit.Parallel.${TEST_NAME}"
    COMMAND
    ${SHELL_EXECUTABLE}
    -c
    "rm -rf ${CHECKPOINT} \
    && ${SPECTRE_TEST_RUNNER} ${CMAKE_BINARY_DIR}/bin/Test_${TEST_NAME} 2>&1 \
    && ${SPECTRE_TEST_RUNNER} ${CMAKE_BINARY_DIR}/bin/Test_${TEST_NAME} \
       +restart ${CHECKPOINT} 2>&1 \
    && rm -rf ${CHECKPOINT}"
    )
  set_standalone_test_properties("Unit.Parallel.${TEST_NAME}")
endfunction()

function(add_algorithm_test BASE_NAME)
  add_standalone_test("Unit.Parallel.${BASE_NAME}" ${ARGN})
  target_link_libraries(
    "Test_${BASE_NAME}"
    PRIVATE
    "${ALGORITHM_TEST_LINK_LIBRARIES}")
endfunction()

# Test GlobalCache
add_algorithm_test("GlobalCache")
add_charm_module(Test_GlobalCache)

add_dependencies(
  module_Test_GlobalCache
  module_GlobalCache
  )

add_dependencies(
  Test_GlobalCache
  module_Test_GlobalCache
  )

add_algorithm_test(
  "AlgorithmBadBoxApply"
  REGEX_TO_MATCH
  "Cannot call apply function of 'error_size_zero' with DataBox \
type 'db::DataBox<brigand::list<")
add_algorithm_test("AlgorithmCore")
add_algorithm_test("AlgorithmLocalSyncAction")
add_algorithm_test(
  "AlgorithmNestedApply1"
  REGEX_TO_MATCH
  "Already performing an Action and cannot execute additional Actions \
from inside of an Action. This is only possible if the simple_action \
function is not invoked via a proxy, which we do not allow.")
add_algorithm_test(
  "AlgorithmNestedApply2"
  REGEX_TO_MATCH
  "Already performing an Action and cannot execute additional Actions \
from inside of an Action. This is only possible if the simple_action \
function is not invoked via a proxy, which we do not allow.")
add_algorithm_test("AlgorithmNodelock")
add_algorithm_test("AlgorithmParallel")
add_algorithm_test("AlgorithmReduction")
add_algorithm_test("PhaseChangeMain")
add_algorithm_test(
  "SectionReductions"
  REGEX_TO_MATCH
  "Element 0 received reduction: Counted 2 odd elements.")

# Charm 7.0.0 has different output in the following test.
if(CHARM_VERSION VERSION_EQUAL 7.0.0)
  message(STATUS "Periodic load balancing with the RotateLB is not working \
correctly with Charm++ 7.0.0. Disabling the test for now until this is \
resolved.")
else()
  add_algorithm_test_with_balancing("AlgorithmParallelWithBalancing"
    "AlgorithmParallel" "Objects migrating: 15")
endif()

add_algorithm_test(
  "AlgorithmGlobalCache"
  INPUT_FILE "Test_AlgorithmGlobalCache.yaml")
add_algorithm_test(
  "AlgorithmPhaseControl"
  INPUT_FILE "Test_AlgorithmPhaseControl.yaml")

add_algorithm_test_with_restart_from_checkpoint("CheckpointRestart")

# Tests that do not require their own Chare setup and can work with the
# unit tests
set(LIBRARY "Test_Parallel")

set(LIBRARY_SOURCES
  Test_GlobalCacheDataBox.cpp
  Test_InboxInserters.cpp
  Test_MemoryMonitor.cpp
  Test_NodeLock.cpp
  Test_Parallel.cpp
  Test_ParallelComponentHelpers.cpp
  Test_PupStlCpp11.cpp
  Test_PupStlCpp17.cpp
  Test_ResourceInfo.cpp
  Test_Serialize.cpp
  Test_TypeTraits.cpp
  )

add_subdirectory(Tags)

add_test_library(
  ${LIBRARY}
  "Parallel"
  "${LIBRARY_SOURCES}"
  "IO;Options;SystemUtilities"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
