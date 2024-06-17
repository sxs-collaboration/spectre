# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY MonteCarlo)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CellCrossingTime.cpp
  CellVolume.cpp
  CouplingTermsForPropagation.cpp
  EvolvePackets.cpp
  InverseJacobianInertialToFluidCompute.cpp
  NeutrinoInteractionTable.cpp
  Packet.cpp
  Scattering.cpp
  TemplatedLocalFunctions.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CellCrossingTime.hpp
  CellVolume.hpp
  CouplingTermsForPropagation.hpp
  EmitPackets.tpp
  EvolvePackets.hpp
  EvolvePacketsInElement.tpp
  ImplicitMonteCarloCorrections.tpp
  InverseJacobianInertialToFluidCompute.hpp
  NeutrinoInteractionTable.hpp
  Packet.hpp
  Scattering.hpp
  TakeTimeStep.tpp
  TemplatedLocalFunctions.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Domain
  ErrorHandling
  GeneralRelativity
  H5
  Hydro
  Options
  Utilities
  PRIVATE
  )
