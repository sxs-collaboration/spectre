# Distributed under the MIT License.
# See LICENSE.txt for details.

set(APPARENT_HORIZONS_TESTS
    ApparentHorizons/Test_SpherepackIterator.cpp
    ApparentHorizons/Test_Strahlkorper.cpp
    ApparentHorizons/Test_YlmSpherepack.cpp
    ApparentHorizons/YlmTestFunctions.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${APPARENT_HORIZONS_TESTS}" PARENT_SCOPE)
