# Distributed under the MIT License.
# See LICENSE.txt for details.

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
  ${SPECTRE_CORE_LIBRARIES}
  ${SPECTRE_LIBRARIES}
  )

spectre_add_catch_tests(Test_ConstGlobalCache)

set(PARALLEL_TESTS
  Parallel/Test_Parallel.cpp
  Parallel/Test_PupStlCpp11.cpp
  Parallel/Test_TypeTraits.cpp
  )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${PARALLEL_TESTS}" PARENT_SCOPE)
