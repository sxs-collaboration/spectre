# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  DarkEnergyFluid.cpp
  Enthalpy.cpp
  HybridEos.cpp
  IdealFluid.cpp
  PolytropicFluid.cpp
  Spectral.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DarkEnergyFluid.hpp
  EquationOfState.hpp
  Factory.hpp
  HybridEos.hpp
  IdealFluid.hpp
  PolytropicFluid.hpp
  Spectral.hpp
  Enthalpy.hpp
  )

add_subdirectory(Python)
