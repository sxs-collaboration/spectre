# Distributed under the MIT License.
# See LICENSE.txt for details.

# Test_AlgorithmParallel
add_executable(
  Test_AlgorithmParallel
  Test_AlgorithmParallel.cpp
  )

add_dependencies(
  Test_AlgorithmParallel
  module_ConstGlobalCache
  module_Main
  )

target_link_libraries(
  Test_AlgorithmParallel
  ErrorHandling
  Informer
  ${SPECTRE_LIBRARIES}
  )

# Test_AlgorithmNodelock
add_executable(
  Test_AlgorithmNodelock
  Test_AlgorithmNodelock.cpp
  )

add_dependencies(
  Test_AlgorithmNodelock
  module_ConstGlobalCache
  module_Main
  )

target_link_libraries(
  Test_AlgorithmNodelock
  ErrorHandling
  Informer
  ${SPECTRE_LIBRARIES}
  )

# Test_AlgorithmReduction
add_executable(
  Test_AlgorithmReduction
  Test_AlgorithmReduction.cpp
  )

add_dependencies(
  Test_AlgorithmReduction
  module_ConstGlobalCache
  module_Main
  )

target_link_libraries(
  Test_AlgorithmReduction
  ErrorHandling
  Informer
  ${SPECTRE_LIBRARIES}
  )

# Test ConstGlobalCache
add_charm_module(Test_ConstGlobalCache)

add_dependencies(
  module_Test_ConstGlobalCache
  module_ConstGlobalCache
  )

add_executable(
  Test_ConstGlobalCache
  Test_ConstGlobalCache.cpp
  )

add_dependencies(
  Test_ConstGlobalCache
  module_Test_ConstGlobalCache
  )

target_link_libraries(
  Test_ConstGlobalCache
  ErrorHandling
  ${SPECTRE_LIBRARIES}
  )

spectre_add_catch_tests(Test_ConstGlobalCache)

add_test(NAME "\"Integration.Parallel.Algorithm\""
  COMMAND ${CMAKE_BINARY_DIR}/bin/Test_AlgorithmParallel)

set_tests_properties(
  "\"Integration.Parallel.Algorithm\"" PROPERTIES
  TIMEOUT 10
  LABELS "integration"
  ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

add_test(NAME "\"Integration.Parallel.Reduction\""
  COMMAND ${CMAKE_BINARY_DIR}/bin/Test_AlgorithmReduction)

set_tests_properties(
  "\"Integration.Parallel.Reduction\"" PROPERTIES
  TIMEOUT 10
  LABELS "integration"
  ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

add_test(NAME "\"Integration.Parallel.Nodelock\""
  COMMAND ${CMAKE_BINARY_DIR}/bin/Test_AlgorithmNodelock)

set_tests_properties(
  "\"Integration.Parallel.Nodelock\"" PROPERTIES
  TIMEOUT 10
  LABELS "integration"
  ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

# Tests that do not require their own Chare setup and can work with the
# unit tests
set(PARALLEL_TESTS
  Parallel/Test_Algorithm.cpp
  Parallel/Test_Parallel.cpp
  Parallel/Test_ParallelComponentHelpers.cpp
  Parallel/Test_PupStlCpp11.cpp
  Parallel/Test_TypeTraits.cpp
  )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${PARALLEL_TESTS}" PARENT_SCOPE)
