# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Barotropic3D.cpp
  DarkEnergyFluid.cpp
  Enthalpy.cpp
  Equilibrium3D.cpp
  HybridEos.cpp
  IdealFluid.cpp
  PiecewisePolytropicFluid.cpp
  PolytropicFluid.cpp
  RegisterDerivedWithCharm.cpp
  Spectral.cpp
  Tabulated3d.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Barotropic3D.hpp
  DarkEnergyFluid.hpp
  Enthalpy.hpp
  Equilibrium3D.hpp
  EquationOfState.hpp
  Factory.hpp
  HybridEos.hpp
  IdealFluid.hpp
  PiecewisePolytropicFluid.hpp
  PolytropicFluid.hpp
  RegisterDerivedWithCharm.hpp
  Spectral.hpp
  Tabulated3d.hpp
  )

add_subdirectory(Python)
