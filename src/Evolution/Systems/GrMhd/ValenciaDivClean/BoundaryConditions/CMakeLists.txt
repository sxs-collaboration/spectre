# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  BoundaryCondition.cpp
  DemandOutgoingCharSpeeds.cpp
  DirichletAnalytic.cpp
  HydroFreeOutflow.cpp
  Reflective.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  DemandOutgoingCharSpeeds.hpp
  DirichletAnalytic.hpp
  Factory.hpp
  HydroFreeOutflow.hpp
  Reflective.hpp
  )
