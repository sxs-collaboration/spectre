# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY LinearOperators)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  CoefficientTransforms.cpp
  DefiniteIntegral.cpp
  Divergence.cpp
  ExponentialFilter.cpp
  IndefiniteIntegral.cpp
  Linearize.cpp
  MeanValue.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CoefficientTransforms.hpp
  DefiniteIntegral.hpp
  Divergence.hpp
  Divergence.tpp
  ExponentialFilter.hpp
  FilterAction.hpp
  IndefiniteIntegral.hpp
  Linearize.hpp
  MeanValue.hpp
  PartialDerivatives.hpp
  PartialDerivatives.tpp
  Tags.hpp
  WeakDivergence.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  DataStructures
  ErrorHandling
  Options
  Spectral
  Utilities
  INTERFACE
  Domain
  DomainStructure
  )
