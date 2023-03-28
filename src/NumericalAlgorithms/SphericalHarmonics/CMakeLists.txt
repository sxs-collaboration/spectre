# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SphericalHarmonics)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ChangeCenterOfStrahlkorper.cpp
  RealSphericalHarmonics.cpp
  SpherepackIterator.cpp
  Strahlkorper.cpp
  StrahlkorperFunctions.cpp
  Tags.cpp
  YlmSpherepack.cpp
  YlmSpherepackHelper.cpp
  YlmToStf.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ChangeCenterOfStrahlkorper.hpp
  RealSphericalHarmonics.hpp
  SpherepackIterator.hpp
  Strahlkorper.hpp
  StrahlkorperFunctions.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TagsTypeAliases.hpp
  YlmSpherepack.hpp
  YlmSpherepackHelper.hpp
  YlmToStf.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  Boost::boost
  DataStructures
  ErrorHandling
  SPHEREPACK
  Utilities
  PRIVATE
  Interpolation
  )
