# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GhValenciaDivClean)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  StressEnergy.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  System.hpp
  StressEnergy.hpp
  Tags.hpp
  TimeDerivativeTerms.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  GeneralRelativity
  INTERFACE
  Domain
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
