# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_ScalarWave)

set(LIBRARY_SOURCES
  BoundaryConditions/Test_ConstraintPreservingSphericalRadiation.cpp
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_Periodic.cpp
  BoundaryConditions/Test_SphericalRadiation.cpp
  BoundaryCorrections/Test_UpwindPenalty.cpp
  Test_Characteristics.cpp
  Test_Constraints.cpp
  Test_EnergyDensity.cpp
  Test_Equations.cpp
  Test_MomentumDensity.cpp
  Test_Tags.cpp
  Test_TimeDerivative.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  MathFunctions
  ScalarWave
  Time
  Utilities
  WaveEquationSolutions
  )
