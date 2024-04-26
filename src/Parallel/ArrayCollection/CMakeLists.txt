# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Tags)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DgElementArrayMemberBase.hpp
  IsDgElementArrayMember.hpp
  IsDgElementCollection.hpp
  ReceiveDataForElement.hpp
  SetTerminateOnElement.hpp
)

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DgElementArrayMemberBase.cpp
)
