# Distributed under the MIT License.
# See LICENSE.txt for details.

# Unit test executables can use this module if they want to run in a Charm++
# environment. They just have to add `WITH_CHARM` when calling
# `add_test_library`.
add_charm_module(TestMainCharm)

# Set include directory for all targets in this directory
include_directories(${CMAKE_SOURCE_DIR}/tests/Unit)
# Charm++ generated headers are created in the build directory
# (by `add_charm_module`)
include_directories(SYSTEM ${CMAKE_BINARY_DIR}/tests/Unit)

add_subdirectory(ControlSystem)
add_subdirectory(DataStructures)
add_subdirectory(Domain)
add_subdirectory(Elliptic)
add_subdirectory(ErrorHandling)
add_subdirectory(Executables)
add_subdirectory(Evolution)
add_subdirectory(Framework)
add_subdirectory(Helpers)
add_subdirectory(IO)
add_subdirectory(Informer)
add_subdirectory(NumericalAlgorithms)
add_subdirectory(Options)
add_subdirectory(Parallel)
add_subdirectory(ParallelAlgorithms)
add_subdirectory(PointwiseFunctions)
add_subdirectory(ArchitectureVectorization)
add_subdirectory(Time)
add_subdirectory(Utilities)
add_subdirectory(Visualization)

# Setup code coverage for unit tests
if(COVERAGE)
  # Setup unit test coverage target.
  setup_target_for_coverage(
      Unit
      ${CMAKE_BINARY_DIR}/docs/html
      unit-test-coverage
      ${CMAKE_CTEST_COMMAND}
      DEPENDS unit-tests
      TESTRUNNER_ARGS
      -j2 -L unit --output-on-failure --repeat after-timeout:3
      IGNORE_COV
      '${CMAKE_BINARY_DIR}/Informer/InfoAtLink.cpp'
      '${CMAKE_SOURCE_DIR}/tests/*'
      # The functions in CharmCompatibility.cpp are not intended to be called
      '${CMAKE_SOURCE_DIR}/src/PythonBindings/CharmCompatibility.cpp'
      # Ignore generated source files in build directory in coverage reporting
      '${CMAKE_BINARY_DIR}/*'
  )
endif()

option(
  UNIT_TESTS_IN_TEST_EXECUTABLES
  "Build unit-tests as part of test-executables"
  ON
  )

if (UNIT_TESTS_IN_TEST_EXECUTABLES)
  add_dependencies(test-executables unit-tests)
endif (UNIT_TESTS_IN_TEST_EXECUTABLES)
