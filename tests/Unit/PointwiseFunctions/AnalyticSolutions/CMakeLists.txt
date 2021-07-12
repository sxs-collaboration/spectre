# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_AnalyticSolutions")

set(LIBRARY_SOURCES
  Test_Tags.cpp
  )

add_subdirectory(Burgers)
add_subdirectory(Elasticity)
add_subdirectory(GeneralRelativity)
add_subdirectory(GhGrMhd)
add_subdirectory(GhRelativisticEuler)
add_subdirectory(GrMhd)
add_subdirectory(Hydro)
add_subdirectory(NewtonianEuler)
add_subdirectory(Poisson)
add_subdirectory(RadiationTransport)
add_subdirectory(RelativisticEuler)
add_subdirectory(ScalarAdvection)
add_subdirectory(WaveEquation)
add_subdirectory(Xcts)

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions"
  "${LIBRARY_SOURCES}"
  "AnalyticSolutions;DataStructures;DataStructuresHelpers"
  )
