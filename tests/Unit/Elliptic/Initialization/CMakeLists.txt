# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticInitialization")

set(LIBRARY_SOURCES
  Test_BoundaryConditions.cpp
  Test_Derivatives.cpp
  Test_DiscontinuousGalerkin.cpp
  Test_Domain.cpp
  Test_Interface.cpp
  Test_LinearSolver.cpp
  Test_Source.cpp
  Test_System.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Initialization/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;ErrorHandling;LinearOperators;Spectral"
  )
