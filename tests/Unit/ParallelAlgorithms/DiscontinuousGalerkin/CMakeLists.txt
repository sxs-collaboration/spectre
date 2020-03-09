# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelDiscontinuousGalerkin")

set(LIBRARY_SOURCES
  Test_CollectDataForFluxes.cpp
  Test_FluxCommunication.cpp
  Test_FluxCommunicationLts.cpp
  Test_HasReceivedFromAllMortars.cpp
  Test_InitializeDomain.cpp
  Test_InitializeInterfaces.cpp
  Test_InitializeMortars.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "CoordinateMaps;DataStructures;Domain;DiscontinuousGalerkin;ErrorHandling;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
