# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelDg)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CollectDataForFluxes.hpp
  FluxCommunication.hpp
  InitializeInterfaces.hpp
  InitializeMortars.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  Domain
  DomainStructure
  ErrorHandling
  Spectral
  Utilities
  )
