# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelArrayCollection)

add_spectre_library(${LIBRARY})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CreateElementCollection.hpp
  CreateElementsUsingDistribution.hpp
  DgElementArrayMember.hpp
  DgElementArrayMemberBase.hpp
  DgElementCollection.hpp
  IsDgElementArrayMember.hpp
  IsDgElementCollection.hpp
  PerformAlgorithmOnElement.hpp
  ReceiveDataForElement.hpp
  SendDataToElement.hpp
  SetTerminateOnElement.hpp
  SimpleActionOnElement.hpp
  SpawnInitializeElementsInCollection.hpp
  StartPhaseOnNodegroup.hpp
  TransformPdalForNodegroup.hpp
)

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DgElementArrayMemberBase.cpp
)

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DomainStructure
  ErrorHandling
  Parallel
  Serialization
  Utilities
  INTERFACE
  Domain
  DomainCreators
  )

add_subdirectory(Tags)
