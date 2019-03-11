# Distributed under the MIT License.
# See LICENSE.txt for details.

# Set up targets to be able to build the tests in stages on TravisCI.

# Generate a list of all the Test_ libraries
get_target_property(LIBS_TO_LINK RunTests LINK_LIBRARIES)

add_custom_target(test-libs-stage1)
add_dependencies(test-libs-stage1
  Test_ApparentHorizons
  Test_ControlSystem
  Test_Amr
  Test_CoordinateMaps
  Test_DomainCreators
  Test_Domain
  Test_Elasticity
  Test_Poisson
  Test_Xcts
  Test_ErrorHandling
  Test_EvolutionActions
  Test_EvolutionConservative
  Test_SlopeLimiters
  Test_EvolutionDiscontinuousGalerkin
  Test_EvolutionEventsAndTriggers
  Test_Burgers
  Test_CurvedScalarWave
  Test_GeneralizedHarmonic
  Test_ValenciaDivClean
  Test_NewtonianEulerSources
  Test_NewtonianEuler
  Test_Valencia
  Test_ScalarWave
  Test_IO
  Test_NumericalFluxes
  Test_NumericalDiscontinuousGalerkin
  Test_NumericalInterpolation
  Test_LinearAlgebra
  Test_LinearOperators
  Test_LinearSolver
  Test_RootFinding
  Test_Spectral
  Test_Options
  Test_BurgersSolutions
  Test_NewtonianEulerSolutions
  Test_PoissonSolutions
  Test_WaveEquation
  Test_GeneralRelativity
  Test_MathFunctions
  Test_Pypp
  Test_TestUtilities
  Test_Time
  Test_Utilities
  )
