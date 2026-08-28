# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarWave)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ApplyTensorYlmFilter.cpp
  Characteristics.cpp
  Constraints.cpp
  EnergyDensity.cpp
  Equations.cpp
  FilledSpherePowerMonitor.cpp
  MomentumDensity.cpp
  SpectralFilter.cpp
  SphericalShellPowerMonitor.cpp
  TimeDerivative.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyTensorYlmFilter.hpp
  Characteristics.hpp
  Constraints.hpp
  EnergyDensity.hpp
  Equations.hpp
  FilledSpherePowerMonitor.hpp
  Initialize.hpp
  MomentumDensity.hpp
  SphericalShellPowerMonitor.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivative.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainBoundaryConditions
  ErrorHandling
  Events
  Evolution
  LinearOperators
  Options
  Spectral
  SphericalHarmonics
  TensorYlm
  Utilities
  INTERFACE
  Initialization
  Parallel
  Serialization
  PRIVATE
  WaveEquationSolutions
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(Instantiations)
add_subdirectory(Python)
