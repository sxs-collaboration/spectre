# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Actions)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Actions.hpp
  AddComputeTags.hpp
  AddSimpleTags.hpp
  FilterAction.hpp
  FunctionsOfTimeAreReady.hpp
  GetItemFromDistributedObject.hpp
  Goto.hpp
  InitializeItems.hpp
  LimiterActions.hpp
  MutateApply.hpp
  RandomizeVariables.hpp
  SetData.hpp
  TerminatePhase.hpp
  UpdateMessageQueue.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  ErrorHandling
  FunctionsOfTime
  Parallel
  Serialization
  Spectral
  Utilities
  )

add_subdirectory(MemoryMonitor)
