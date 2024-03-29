# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  MonotonicityPreserving5.cpp
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
  Factory.hpp
  FiniteDifference.hpp
  MonotonicityPreserving5.hpp
  MonotonisedCentral.hpp
  PositivityPreservingAdaptiveOrder.hpp
  Reconstructor.hpp
  ReconstructWork.hpp
  ReconstructWork.tpp
  RegisterDerivedWithCharm.hpp
  Tag.hpp
  Wcns5z.hpp
  )
