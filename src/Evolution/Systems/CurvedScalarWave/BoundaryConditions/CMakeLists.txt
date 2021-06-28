# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ConstraintPreservingSphericalRadiation.cpp
  BoundaryCondition.cpp
  Outflow.cpp
  RegisterDerivedWithCharm.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ConstraintPreservingSphericalRadiation.hpp
  BoundaryCondition.hpp
  Factory.hpp
  Outflow.hpp
  RegisterDerivedWithCharm.hpp
)
