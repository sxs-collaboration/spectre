# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_MonteCarlo")

set(LIBRARY_SOURCES
  Test_CellCrossingTime.cpp
  Test_CellVolume.cpp
  Test_CommunicationTags.cpp
  Test_EmitPackets.cpp
  Test_EvolvePackets.cpp
  Test_ImplicitMonteCarloCorrections.cpp
  Test_InterpolateOpacities.cpp
  Test_InverseJacobianInertialToFluid.cpp
  Test_NeutrinoInteractionTable.cpp
  Test_Packet.cpp
  Test_Scattering.cpp
  Test_TakeTimeStep.cpp
  Test_TimeStepAction.cpp
  )

add_test_library(
  ${LIBRARY}
  "${LIBRARY_SOURCES}"
  ""
  )
target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DgSubcell
  GeneralRelativity
  GeneralRelativityHelpers
  GeneralRelativitySolutions
  H5
  Hydro
  HydroHelpers
  Informer
  MonteCarlo
  Time
  )
