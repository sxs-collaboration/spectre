# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativity)

set(LIBRARY_SOURCES
    Christoffel.cpp
    ComputeGhQuantities.cpp
    ComputeSpacetimeQuantities.cpp
    IndexManipulation.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})
