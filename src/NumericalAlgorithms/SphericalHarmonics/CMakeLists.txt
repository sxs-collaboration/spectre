# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SphericalHarmonics)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AngularOrdering.cpp
  ChangeCenterOfStrahlkorper.cpp
  InitialShape.cpp
  RealSphericalHarmonics.cpp
  SpherepackIterator.cpp
  Strahlkorper.cpp
  StrahlkorperFunctions.cpp
  Spherepack.cpp
  SpherepackCache.cpp
  SpherepackHelper.cpp
  Tags.cpp
  YlmToStf.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AngularOrdering.hpp
  ChangeCenterOfStrahlkorper.hpp
  InitialShape.hpp
  RealSphericalHarmonics.hpp
  SpherepackIterator.hpp
  Strahlkorper.hpp
  StrahlkorperFunctions.hpp
  Spherepack.hpp
  SpherepackCache.hpp
  SpherepackHelper.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TagsTypeAliases.hpp
  YlmToStf.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  BLAS::BLAS
  Boost::boost
  DataStructures
  ErrorHandling
  slatec
  SPHEREPACK
  Utilities
  PRIVATE
  FiniteDifference
  Interpolation
  )

add_subdirectory(IO)
add_subdirectory(Python)
