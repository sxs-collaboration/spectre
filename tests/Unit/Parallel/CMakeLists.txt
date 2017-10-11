# Distributed under the MIT License.
# See LICENSE.txt for details.

add_charm_module(ParallelTestChares)

add_charm_module(Test_ConstGlobalCache)

add_charm_module(Test_Main)

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

add_dependencies(
  module_Test_Main
  module_ConstGlobalCache
  module_Main
  module_ParallelTestChares
  )

add_executable(
  Test_Main
  Test_Main.cpp
  Test_Parallel.cpp
  Test_PupStlCpp11.cpp
  Test_TypeTraits.cpp
  )

add_dependencies(
  Test_Main
  module_Test_Main
  )

target_link_libraries(
  Test_Main
  Informer
  Utilities
  ${SPECTRE_LIBRARIES}
  )

spectre_add_catch_tests(Test_Main)
