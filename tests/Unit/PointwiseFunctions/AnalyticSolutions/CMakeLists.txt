# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_AnalyticSolutions")

set(LIBRARY_SOURCES
  Test_AnalyticSolution.cpp
  Test_Tags.cpp
  )

add_subdirectory(Burgers)
add_subdirectory(Elasticity)
add_subdirectory(ForceFree)
add_subdirectory(GeneralRelativity)
add_subdirectory(GhGrMhd)
add_subdirectory(GhRelativisticEuler)
add_subdirectory(GrMhd)
add_subdirectory(Hydro)
add_subdirectory(NewtonianEuler)
add_subdirectory(Poisson)
add_subdirectory(Punctures)
add_subdirectory(RadiationTransport)
add_subdirectory(RelativisticEuler)
add_subdirectory(ScalarAdvection)
add_subdirectory(WaveEquation)
add_subdirectory(Xcts)

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  AnalyticSolutions
  DataStructures
  DataStructuresHelpers
)
