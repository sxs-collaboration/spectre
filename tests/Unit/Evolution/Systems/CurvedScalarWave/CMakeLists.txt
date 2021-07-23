# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_CurvedScalarWave")

set(LIBRARY_SOURCES
  BoundaryCorrections/Test_UpwindPenalty.cpp
  Test_Constraints.cpp
  Test_Characteristics.cpp
  Test_Tags.cpp
  Test_TimeDerivative.cpp
  Test_UpwindPenaltyCorrection.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/CurvedScalarWave/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  CurvedScalarWave
  CurvedScalarWaveHelpers
  GeneralRelativityHelpers
  Utilities
  )
