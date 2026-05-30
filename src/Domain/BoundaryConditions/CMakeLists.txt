# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainBoundaryConditions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Cartoon.cpp
  None.cpp
  Periodic.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCondition.hpp
  Cartoon.hpp
  GetBoundaryConditionsBase.hpp
  None.hpp
  Periodic.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Options
  Serialization
  Spectral
  Utilities
  INTERFACE
  ErrorHandling
  )
