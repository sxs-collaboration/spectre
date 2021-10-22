# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SphericalHarmonics)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  SpherepackIterator.cpp
  YlmSpherepack.cpp
  YlmSpherepackHelper.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  SpherepackIterator.hpp
  YlmSpherepack.hpp
  YlmSpherepackHelper.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Blas
  DataStructures
  ErrorHandling
  SPHEREPACK
  Utilities
  )
