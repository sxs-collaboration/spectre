# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY MonteCarlo)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  EvolveMonteCarloPackets.cpp
  InverseJacobianInertialToFluidCompute.cpp
  MonteCarloPacket.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  EvolveMonteCarloPackets.hpp
  InverseJacobianInertialToFluidCompute.hpp
  MonteCarloPacket.hpp
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
