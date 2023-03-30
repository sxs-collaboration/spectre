# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FiniteDifference)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AoWeno.cpp
  DerivativeOrder.cpp
  FallbackReconstructorType.cpp
  Filter.cpp
  Minmod.cpp
  MonotonicityPreserving5.cpp
  MonotonisedCentral.cpp
  NonUniform1D.cpp
  PartialDerivatives.cpp
  PositivityPreservingAdaptiveOrder.cpp
  Unlimited.cpp
  Wcns5z.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AoWeno.hpp
  DerivativeOrder.hpp
  FallbackReconstructorType.hpp
  Filter.hpp
  FiniteDifference.hpp
  HighOrderFluxCorrection.hpp
  Minmod.hpp
  MonotonicityPreserving5.hpp
  MonotonisedCentral.hpp
  NeighborDataAsVariables.hpp
  NonUniform1D.hpp
  PartialDerivatives.hpp
  PartialDerivatives.tpp
  PositivityPreservingAdaptiveOrder.hpp
  Reconstruct.hpp
  Reconstruct.tpp
  Unlimited.hpp
  Wcns5z.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Utilities
  PRIVATE
  DataStructures
  DomainStructure
  ErrorHandling
  )
