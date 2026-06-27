# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SphericalHarmonics)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AngularOrdering.cpp
  RealSphericalHarmonics.cpp
  SpherepackIterator.cpp
  Spherepack.cpp
  SpherepackCache.cpp
  SpherepackHelper.cpp
  YlmToStf.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AngularOrdering.hpp
  RealSphericalHarmonics.hpp
  SpherepackIterator.hpp
  Spherepack.hpp
  SpherepackCache.hpp
  SpherepackHelper.hpp
  YlmToStf.hpp
  )

# This library is intentionally low-level: it holds only the Spherepack-based
# spherical-harmonic implementation and it must NOT depend on Spectral (directly
# or transitively). Higher-level functionality that depends on Spectral lives in
# the Strahlkorper and TensorYlm libraries.
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
  )

add_subdirectory(Python)
