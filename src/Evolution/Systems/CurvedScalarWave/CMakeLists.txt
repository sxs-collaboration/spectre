# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CurvedScalarWave)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Characteristics.cpp
  Constraints.cpp
  TimeDerivative.cpp
  VolumeTermsInstantiation.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BackgroundSpacetime.hpp
  Characteristics.hpp
  CalculateGrVars.hpp
  Constraints.hpp
  Initialize.hpp
  PsiSquared.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TimeDerivative.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Domain
  ErrorHandling
  GeneralRelativity
  Serialization
  Utilities
  )

add_subdirectory(BoundaryConditions)
add_subdirectory(BoundaryCorrections)
add_subdirectory(Worldtube)
