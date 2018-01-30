# Distributed under the MIT License.
# See LICENSE.txt for details.

set(executable RunTests)

set(SPECTRE_TESTS "TestFramework.cpp")

set(SPECTRE_COMPILATION_TESTS "TestCompilationFramework.cpp")

add_subdirectory(ApparentHorizons)
add_subdirectory(ControlSystem)
add_subdirectory(DataStructures)
add_subdirectory(Domain)
add_subdirectory(ErrorHandling)
add_subdirectory(Evolution)
add_subdirectory(IO)
add_subdirectory(NumericalAlgorithms)
add_subdirectory(Options)
add_subdirectory(Parallel)
add_subdirectory(PointwiseFunctions)
add_subdirectory(Time)
add_subdirectory(Utilities)

set(SPECTRE_TESTS "${SPECTRE_TESTS};Test_TestHelpers.cpp")

add_charm_module(${executable})

add_executable(
    ${executable}
    ${executable}.cpp
    ${executable}.decl.h
    ${executable}.def.h
    ${SPECTRE_TESTS}
)

target_link_libraries(
  ${executable}
  ApparentHorizons
  ControlSystem
  CoordinateMaps
  CurvedScalarWave
  Domain
  DomainCreators
  EinsteinSolutions
  GeneralizedHarmonic
  GeneralRelativity
  IO
  LinearOperators
  RootFinding
  ScalarWave
  Time
  WaveEquation
  ${SPECTRE_LIBRARIES}
)

add_dependencies(
  ${executable}
  module_ConstGlobalCache
  module_Main
  )

spectre_add_catch_tests(${executable})

# Setup code coverage for unit tests
if(COVERAGE)
  # Setup unit test coverage target.
  setup_target_for_coverage(
      Unit
      ${CMAKE_BINARY_DIR}/docs/html
      unit-test-coverage
      ${CMAKE_CTEST_COMMAND}
      DEPENDS ${executable}
      TESTRUNNER_ARGS
      "--output-on-failure"
      IGNORE_COV
      '${CMAKE_BINARY_DIR}/Informer/InfoFromBuild.cpp'
      '${CMAKE_SOURCE_DIR}/tests/*'
  )
endif()

# Setup tests of static_assert's and other compilation failures

set(EXECUTABLE compilation-failure-tests)

add_executable(${EXECUTABLE} ${SPECTRE_COMPILATION_TESTS})

set_target_properties(
  ${EXECUTABLE}
  PROPERTIES
  COMPILE_FLAGS "\${WHICH_TEST} -DFILE_IS_COMPILATION_TEST=\"static_assert(false, \\\"File is compilation test\\\");\""
  EXCLUDE_FROM_ALL TRUE
  )

add_compilation_tests(${EXECUTABLE})
