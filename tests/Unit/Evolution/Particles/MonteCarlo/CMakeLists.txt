# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_MonteCarlo")

set(LIBRARY_SOURCES
  Test_EmitPackets.cpp
  Test_EvolvePackets.cpp
  Test_InterpolateOpacities.cpp
  Test_InverseJacobianInertialToFluid.cpp
  Test_Packet.cpp
  Test_Scattering.cpp
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
  GeneralRelativity
  GeneralRelativityHelpers
  Hydro
  HydroHelpers
  MonteCarlo
  )
