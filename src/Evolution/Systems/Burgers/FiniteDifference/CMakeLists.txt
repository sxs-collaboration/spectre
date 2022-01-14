# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  MonotonisedCentral.cpp
  Reconstructor.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryConditionGhostData.hpp
  Factory.hpp
  FiniteDifference.hpp
  MonotonisedCentral.hpp
  Reconstructor.hpp
  ReconstructWork.hpp
  ReconstructWork.tpp
  RegisterDerivedWithCharm.hpp
  Tags.hpp
  )
