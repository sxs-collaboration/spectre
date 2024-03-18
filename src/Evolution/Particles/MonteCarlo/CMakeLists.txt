# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY MonteCarlo)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  EvolvePackets.cpp
  InverseJacobianInertialToFluidCompute.cpp
  TemplatedLocalFunctions.cpp
  Packet.cpp
  Scattering.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  EmitPackets.tpp
  EvolvePackets.hpp
  EvolvePacketsInElement.tpp
  InverseJacobianInertialToFluidCompute.hpp
  Packet.hpp
  Scattering.hpp
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
  Hydro
  Options
  Utilities
  PRIVATE
  )
