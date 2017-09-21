# Distributed under the MIT License.
# See LICENSE.txt for details.

set(MY_LIBRARY_SOURCES
    Spectral/DefiniteIntegral.cpp
    Spectral/LegendreGaussLobatto.cpp
    Spectral/Linearize.cpp
    Spectral/MeanValue.cpp
    )

set(LIBRARY_SOURCES "${LIBRARY_SOURCES};${MY_LIBRARY_SOURCES}" PARENT_SCOPE)
