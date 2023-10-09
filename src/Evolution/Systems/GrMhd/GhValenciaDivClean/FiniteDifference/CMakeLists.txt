# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Derivatives.cpp
  FilterOptions.cpp
  Filters.cpp
  MonotonisedCentral.cpp
  PositivityPreservingAdaptiveOrder.cpp
  Reconstructor.cpp
  RegisterDerivedWithCharm.cpp
  Wcns5z.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryConditionGhostData.hpp
  Derivatives.hpp
  Factory.hpp
  FilterOptions.hpp
  Filters.hpp
  FiniteDifference.hpp
  MonotonisedCentral.hpp
  PositivityPreservingAdaptiveOrder.hpp
  ReconstructWork.hpp
  ReconstructWork.tpp
  Reconstructor.hpp
  RegisterDerivedWithCharm.hpp
  Tag.hpp
  Wcns5z.hpp
  )
