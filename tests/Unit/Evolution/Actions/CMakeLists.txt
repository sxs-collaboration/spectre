# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EvolutionActions")

set(LIBRARY_SOURCES
  Test_AddMeshVelocityNonconservative.cpp
  Test_AddMeshVelocitySourceTerms.cpp
  Test_ComputeTimeDerivative.cpp
  Test_ComputeVolumeFluxes.cpp
  Test_ComputeVolumeSources.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Actions/"
  "${LIBRARY_SOURCES}"
  "Boost::boost;DataStructures;Domain;ErrorHandling"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
