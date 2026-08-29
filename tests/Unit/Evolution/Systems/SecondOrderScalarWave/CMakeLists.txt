# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_SecondOrderScalarWave)

set(LIBRARY_SOURCES
  BoundaryConditions/Test_Periodic.cpp
  Test_Characteristics.cpp
  Test_System.cpp
  Test_Tags.cpp
  Test_TimeDerivative.cpp
  Test_UpdateAuxiliaryVariables.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  LinearOperators
  MathFunctions
  SecondOrderScalarWave
  Spectral
  Utilities
  WaveEquationSolutions
  )
