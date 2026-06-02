# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Strahlkorper)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ChangeCenterOfStrahlkorper.cpp
  InitialShape.cpp
  Strahlkorper.cpp
  StrahlkorperFunctions.cpp
  Tags.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ChangeCenterOfStrahlkorper.hpp
  InitialShape.hpp
  Strahlkorper.hpp
  StrahlkorperFunctions.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TagsTypeAliases.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  BLAS::BLAS
  Boost::boost
  DataStructures
  ErrorHandling
  SphericalHarmonics
  Utilities
  PRIVATE
  FiniteDifference
  LinearAlgebra
  )

add_subdirectory(IO)
add_subdirectory(Python)
