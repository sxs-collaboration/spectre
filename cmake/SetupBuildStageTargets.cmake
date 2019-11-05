# Distributed under the MIT License.
# See LICENSE.txt for details.

# Set up targets to be able to build the tests in stages on TravisCI.

# Generate a list of all the Test_ libraries
get_target_property(LIBS_TO_LINK RunTests LINK_LIBRARIES)

add_custom_target(test-libs-data-structures)
add_custom_target(test-libs-domain)
add_custom_target(test-libs-elliptic)
add_custom_target(test-libs-evolution)
add_custom_target(test-libs-numerical-algorithms)
add_custom_target(test-libs-parallel-algorithms)
add_custom_target(test-libs-pointwise-functions)
add_custom_target(test-libs-other)

add_dependencies(test-libs-data-structures
  Test_DataStructures
  Test_DataBox
  Test_Tensor
  Test_EagerMath
  Test_Expressions
  )

add_dependencies(test-libs-domain
  Test_Domain
  Test_Amr
  Test_CoordinateMaps
  Test_DomainCreators
  )

add_dependencies(test-libs-elliptic
  Test_EllipticActions
  Test_EllipticDG
  Test_Elasticity
  Test_Poisson
  Test_Xcts
  )

add_dependencies(test-libs-evolution
  Test_EvolutionActions
  Test_EvolutionConservative
  Test_Limiters
  Test_EventsAndTriggers
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
  )

add_dependencies(test-libs-numerical-algorithms
  Test_Convergence
  Test_NumericalDiscontinuousGalerkin
  Test_NumericalFluxes
  Test_NumericalInterpolation
  Test_LinearAlgebra
  Test_LinearOperators
  Test_RootFinding
  Test_Spectral
  )

add_dependencies(test-libs-parallel-algorithms
  Test_EventsAndTriggers
  Test_Initialization
  Test_InitializationActions
  Test_ParallelAlgorithmsActions
  Test_ParallelAlgorithmsEvents
  Test_ParallelConjugateGradient
  Test_ParallelDiscontinuousGalerkin
  Test_ParallelGmres
  Test_ParallelLinearSolver
  )

add_dependencies(test-libs-pointwise-functions
  Test_GrMhdAnalyticData
  Test_BurgersSolutions
  Test_ElasticitySolutions
  Test_EquationsOfState
  Test_GeneralRelativitySolutions
  Test_GrMhdSolutions
  Test_NewtonianEulerSolutions
  Test_PoissonSolutions
  Test_M1GreySolutions
  Test_RelativisticEulerSolutions
  Test_WaveEquation
  Test_XctsSolutions
  Test_ConstitutiveRelations
  Test_GeneralRelativity
  Test_Hydro
  Test_MathFunctions
  )

add_dependencies(test-libs-other
  Test_ApparentHorizons
  Test_ControlSystem
  Test_ErrorHandling
  Test_IO
  Test_Informer
  Test_Options
  Test_Pypp
  Test_TestUtilities
  Test_Time
  Test_Utilities
  )
