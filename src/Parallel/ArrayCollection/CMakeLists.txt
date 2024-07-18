# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Tags)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CreateElementCollection.hpp
  DgElementArrayMember.hpp
  DgElementArrayMemberBase.hpp
  DgElementCollection.hpp
  IsDgElementArrayMember.hpp
  IsDgElementCollection.hpp
  ReceiveDataForElement.hpp
  SendDataToElement.hpp
  SetTerminateOnElement.hpp
  SpawnInitializeElementsInCollection.hpp
  StartPhaseOnNodegroup.hpp
  TransformPdalForNodegroup.hpp
)

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DgElementArrayMemberBase.cpp
)
