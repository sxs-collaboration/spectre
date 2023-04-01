# Distributed under the MIT License.
# See LICENSE.txt for details.


spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  TimeDerivative.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ChangeSlabSize.hpp
  InitializeElementFacesGridCoordinates.hpp
  InitializeEvolvedVariables.hpp
  ReceiveElementData.hpp
  SendToElements.hpp
  TimeDerivative.hpp
)
