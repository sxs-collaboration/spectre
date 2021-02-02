# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarWave)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Constraints.cpp
  Equations.cpp
  TimeDerivative.cpp
  UpwindPenaltyCorrection.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Characteristics.hpp
  Constraints.hpp
  Equations.hpp
  Initialize.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivative.hpp
  UpwindPenaltyCorrection.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
