# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY AnalyticSolutions)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticSolution.hpp
  Tags.hpp
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
