# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarAdvection)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Fluxes.cpp
  TimeDerivativeTerms.cpp
  VelocityField.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Fluxes.hpp
  System.hpp
  Tags.hpp
  TimeDerivativeTerms.hpp
  VelocityField.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  DgSubcell
  INTERFACE
  ErrorHandling
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(FiniteDifference)
add_subdirectory(Subcell)
