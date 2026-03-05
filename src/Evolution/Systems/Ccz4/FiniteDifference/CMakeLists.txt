# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SoCcz4)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ApplyFilter.cpp
  Derivatives.cpp
  DummyReconstructor.cpp
  EnforceConstrainedEvolution.cpp
  Filter.cpp
  GhostData.cpp
  Reconstructor.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ApplyFilter.hpp
  BoundaryConditionGhostData.hpp
  Derivatives.hpp
  DummyReconstructor.hpp
  EnforceConstrainedEvolution.hpp
  Filter.hpp
  GhostData.hpp
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
  Options
  Spectral
  Utilities
  INTERFACE
  )

add_subdirectory(../BoundaryConditions ${CMAKE_CURRENT_BINARY_DIR}/BoundaryConditions)
