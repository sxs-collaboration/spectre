# Distributed under the MIT License.
# See LICENSE.txt for details.

add_charm_module(GlobalCache)
add_charm_module(Main)

set(LIBRARY Parallel)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InitializationFunctions.cpp
  NodeLock.cpp
  Reduction.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Algorithm.hpp
  AlgorithmMetafunctions.hpp
  ArrayIndex.hpp
  Callback.hpp
  CharmMain.tpp
  CharmPupable.hpp
  CharmRegistration.hpp
  CreateFromOptions.hpp
  GetSection.hpp
  GlobalCache.hpp
  GlobalCacheDeclare.hpp
  InboxInserters.hpp
  Info.hpp
  InitializationFunctions.hpp
  Invoke.hpp
  Local.hpp
  Main.hpp
  MaxInlineMethodsReached.hpp
  NodeLock.hpp
  ParallelComponentHelpers.hpp
  PhaseControlReductionHelpers.hpp
  PhaseDependentActionList.hpp
  Printf.hpp
  PupStlCpp11.hpp
  PupStlCpp17.hpp
  Reduction.hpp
  ReductionDeclare.hpp
  RegisterDerivedClassesWithCharm.hpp
  ResourceInfo.hpp
  Section.hpp
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
  Charmxx::charmxx
  ErrorHandling
  SystemUtilities
  Utilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  module_Main          # GlobalCache module depends on Main module
  )

add_subdirectory(Actions)
add_subdirectory(Algorithms)
add_subdirectory(MemoryMonitor)
add_subdirectory(PhaseControl)
add_subdirectory(Protocols)
add_subdirectory(Tags)
