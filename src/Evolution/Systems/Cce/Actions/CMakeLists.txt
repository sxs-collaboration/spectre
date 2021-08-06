# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ScriObserveInterpolated.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryComputeAndSendToEvolution.hpp
  CalculateScriInputs.hpp
  CharacteristicEvolutionBondiCalculations.hpp
  FilterSwshVolumeQuantity.hpp
  InitializeCharacteristicEvolutionScri.hpp
  InitializeCharacteristicEvolutionTime.hpp
  InitializeCharacteristicEvolutionVariables.hpp
  InitializeFirstHypersurface.hpp
  InitializeWorldtubeBoundary.hpp
  InsertInterpolationScriData.hpp
  ReceiveGhWorldtubeData.hpp
  ReceiveWorldtubeData.hpp
  RequestBoundaryData.hpp
  ScriObserveInterpolated.hpp
  TimeManagement.hpp
  UpdateGauge.hpp
  )
