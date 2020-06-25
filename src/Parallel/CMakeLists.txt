# Distributed under the MIT License.
# See LICENSE.txt for details.

add_charm_module(ConstGlobalCache)
add_charm_module(Main)

set(LIBRARY Parallel)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Abort.hpp
  Algorithm.hpp
  AlgorithmMetafunctions.hpp
  ArrayIndex.hpp
  CharmMain.tpp
  CharmPupable.hpp
  CharmRegistration.hpp
  ConstGlobalCache.hpp
  CreateFromOptions.hpp
  Exit.hpp
  InboxInserters.hpp
  Info.hpp
  InitializationFunctions.hpp
  Invoke.hpp
  Main.hpp
  NodeLock.hpp
  ParallelComponentHelpers.hpp
  PhaseDependentActionList.hpp
  Printf.hpp
  PupStlCpp11.hpp
  Reduction.hpp
  RegisterDerivedClassesWithCharm.hpp
  Serialize.hpp
  SimpleActionVisitation.hpp
  TypeTraits.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  DataStructures
  ErrorHandling
  Informer
  Options
  Utilities
  )

add_subdirectory(Actions)
