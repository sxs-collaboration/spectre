# Distributed under the MIT License.
# See LICENSE.txt for details.

add_charm_module(GlobalCache)
add_charm_module(Main)

set(LIBRARY Parallel)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  NodeLock.cpp
  )

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
  CreateFromOptions.hpp
  Exit.hpp
  GlobalCache.hpp
  InboxInserters.hpp
  InitializationFunctions.hpp
  Invoke.hpp
  Main.hpp
  NodeLock.hpp
  ParallelComponentHelpers.hpp
  PhaseDependentActionList.hpp
  Printf.hpp
  PupStlCpp11.hpp
  PupStlCpp17.hpp
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
  Informer
  Options
  PUBLIC
  ErrorHandling
  SystemUtilities
  Utilities
  )

add_subdirectory(Actions)
add_subdirectory(Algorithms)
