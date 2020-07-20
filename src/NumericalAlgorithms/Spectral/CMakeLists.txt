# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Spectral)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Chebyshev.cpp
  ComplexDataView.cpp
  Filtering.cpp
  FiniteDifference.cpp
  Legendre.cpp
  Mesh.cpp
  Projection.cpp
  Spectral.cpp
  SwshCoefficients.cpp
  SwshCollocation.cpp
  SwshDerivatives.cpp
  SwshFiltering.cpp
  SwshInterpolation.cpp
  SwshTags.cpp
  SwshTransform.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComplexDataView.hpp
  Filtering.hpp
  Mesh.hpp
  Projection.hpp
  Spectral.hpp
  SwshCoefficients.hpp
  SwshCollocation.hpp
  SwshDerivatives.hpp
  SwshFiltering.hpp
  SwshInterpolation.hpp
  SwshSettings.hpp
  SwshTags.hpp
  SwshTransform.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Libsharp
  PRIVATE
  Blas
  Boost::boost
  Lapack
  )
