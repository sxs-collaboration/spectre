# Distributed under the MIT License.
# See LICENSE.txt for details.


spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  UpdateAcceleration.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ChangeSlabSize.hpp
  InitializeElementFacesGridCoordinates.hpp
  InitializeEvolvedVariables.hpp
  IterateAccelerationTerms.hpp
  ObserveWorldtubeSolution.hpp
  ReceiveElementData.hpp
  SendToElements.hpp
  UpdateAcceleration.hpp
  UpdateFunctionsOfTime.hpp
)
