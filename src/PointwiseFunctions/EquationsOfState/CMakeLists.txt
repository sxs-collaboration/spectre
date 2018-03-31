# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY EquationsOfState)

set(LIBRARY_SOURCES
    Polytrope.cpp
    ConstDensity.cpp
    )

add_library(${LIBRARY} ${LIBRARY_SOURCES})


