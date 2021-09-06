# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_CurvedScalarWave")

set(LIBRARY_SOURCES
  BoundaryCorrections/Test_UpwindPenalty.cpp
  BoundaryConditions/Test_ConstraintPreservingSphericalRadiation.cpp
  BoundaryConditions/Test_Outflow.cpp
  Test_Constraints.cpp
  Test_Characteristics.cpp
  Test_Tags.cpp
  Test_TimeDerivative.cpp
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
  CurvedScalarWave
  CurvedScalarWaveHelpers
  DataStructures
  GeneralRelativityHelpers
  Utilities
  )
