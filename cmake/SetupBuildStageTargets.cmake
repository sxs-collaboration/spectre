# Distributed under the MIT License.
# See LICENSE.txt for details.

# Set up targets to be able to build the tests in stages on TravisCI.

# Generate a list of all the Test_ libraries
get_target_property(LIBS_TO_LINK RunTests LINK_LIBRARIES)

add_custom_target(test-libs-stage1)
add_dependencies(test-libs-stage1
  Test_ApparentHorizons
  Test_ControlSystem
  Test_Domain
  Test_Amr
  Test_CoordinateMaps
  Test_DomainCreators
  Test_EllipticActions
  Test_EllipticDG
  Test_EllipticInitialization
  Test_Elasticity
  Test_Poisson
  Test_PoissonActions
  Test_Xcts
  Test_ErrorHandling
  Test_EvolutionActions
  Test_EvolutionConservative
  Test_EvolutionDiscontinuousGalerkin
  Test_SlopeLimiters
  Test_EvolutionEventsAndTriggers
  Test_Burgers
  Test_CurvedScalarWave
  Test_GeneralizedHarmonic
  Test_GeneralizedHarmonicGaugeSourceFunctions
  Test_ValenciaDivClean
  Test_NewtonianEuler
  Test_NewtonianEulerSources
  Test_M1Grey
  Test_Valencia
  Test_VariableFixing
  Test_ScalarWave
  Test_IO
  Test_Informer
  Test_Convergence
  Test_NumericalDiscontinuousGalerkin
  Test_NumericalFluxes
  Test_NumericalInterpolation
  Test_LinearAlgebra
  Test_LinearOperators
  Test_LinearSolver
  Test_LinearSolverActions
  Test_ConjugateGradientAlgorithm
  Test_DistributedConjugateGradientAlgorithm
  Test_ConjugateGradient
  Test_DistributedGmresAlgorithm
  Test_GmresAlgorithm
  Test_Gmres
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
