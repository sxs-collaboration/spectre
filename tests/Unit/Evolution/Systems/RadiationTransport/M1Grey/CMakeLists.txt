# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_M1Grey")

set(LIBRARY_SOURCES
  BoundaryConditions/Test_DirichletAnalytic.cpp
  BoundaryConditions/Test_Periodic.cpp
  BoundaryCorrections/Test_Rusanov.cpp
  Test_Actions.cpp
  Test_Fluxes.cpp
  Test_M1Closure.cpp
  Test_M1HydroCoupling.cpp
  Test_Sources.cpp
  Test_System.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  DiscontinuousGalerkin
  Domain
  M1Grey
  M1GreySolutions
  Parallel
  Spectral
  Utilities
  )
