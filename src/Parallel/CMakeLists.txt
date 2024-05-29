# Distributed under the MIT License.
# See LICENSE.txt for details.

add_charm_module(GlobalCache)
add_charm_module(Main)

set(LIBRARY Parallel)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ArrayComponentId.cpp
  CharmRegistration.cpp
  InitializationFunctions.cpp
  NodeLock.cpp
  Phase.cpp
  Reduction.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AlgorithmExecution.hpp
  AlgorithmMetafunctions.hpp
  ArrayComponentId.hpp
  ArrayIndex.hpp
  Callback.hpp
  CharmMain.tpp
  CharmRegistration.hpp
  CreateElementsUsingDistribution.hpp
  CreateFromOptions.hpp
  DistributedObject.hpp
  DomainDiagnosticInfo.hpp
  ElementRegistration.hpp
  ExitCode.hpp
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
  OutputInbox.hpp
  ParallelComponentHelpers.hpp
  Phase.hpp
  PhaseControlReductionHelpers.hpp
  PhaseDependentActionList.hpp
  Reduction.hpp
  ReductionDeclare.hpp
  ResourceInfo.hpp
  Section.hpp
  Spinlock.hpp
  StaticSpscQueue.hpp
  TypeTraits.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  Boost::program_options
  DataStructures
  Domain
  Informer
  PUBLIC
  Charmxx::charmxx
  DomainStructure
  ErrorHandling
  Options
  Printf
  Serialization
  SystemUtilities
  Utilities
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  module_Main          # GlobalCache module depends on Main module
  )

add_subdirectory(Algorithms)
add_subdirectory(ArrayCollection)
add_subdirectory(MemoryMonitor)
add_subdirectory(PhaseControl)
add_subdirectory(Printf)
add_subdirectory(Protocols)
add_subdirectory(Tags)
