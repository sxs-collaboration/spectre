# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativity)

set(LIBRARY_SOURCES
    Christoffel.cpp
    ComputeSpacetimeQuantities.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})
