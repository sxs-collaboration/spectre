# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitializeElementFacesGridCoordinates.hpp
  InitializeEvolvedVariables.hpp
  ReceiveElementData.hpp
  SendToElements.hpp
)
