# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GeneralRelativity)

set(LIBRARY_SOURCES
    ComputeSpacetimeQuantities.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})
