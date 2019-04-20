# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Spectral)

set(LIBRARY_SOURCES
    Chebyshev.cpp
    ComplexDataView.cpp
    Filtering.cpp
    Legendre.cpp
    Projection.cpp
    Spectral.cpp
    SwshCoefficients.cpp
    SwshCollocation.cpp
    SwshTags.cpp
    )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
