# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SpinWeightedSphericalHarmonics)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ComplexDataView.cpp
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
  Options
  Spectral
  PRIVATE
  Boost::boost
  )
