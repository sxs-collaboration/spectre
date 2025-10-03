# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SoCcz4)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Derivatives.cpp
  DummyReconstructor.cpp
  Reconstructor.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryConditionGhostData.hpp
  Derivatives.hpp
  DummyReconstructor.hpp
  Reconstructor.hpp
  SoTimeDerivative.hpp
  System.hpp
  Tags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Ccz4
  DataStructures
  DomainStructure
  ErrorHandling
  DgSubcell
  FiniteDifference
  GeneralRelativity
  GeneralRelativitySolutions
  Options
  Spectral
  Utilities
  INTERFACE
  )

add_subdirectory(../BoundaryConditions ${CMAKE_CURRENT_BINARY_DIR}/BoundaryConditions)
