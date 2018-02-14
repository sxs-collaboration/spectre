# Distributed under the MIT License.
# See LICENSE.txt for details.

set(APPARENT_HORIZONS_TESTS
    ApparentHorizons/StrahlkorperGrTestHelpers.cpp
    ApparentHorizons/Test_FastFlow.cpp
    ApparentHorizons/Test_SpherepackIterator.cpp
    ApparentHorizons/Test_Strahlkorper.cpp
    ApparentHorizons/Test_StrahlkorperDataBox.cpp
    ApparentHorizons/Test_StrahlkorperGr.cpp
    ApparentHorizons/Test_YlmSpherepack.cpp
    ApparentHorizons/YlmTestFunctions.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${APPARENT_HORIZONS_TESTS}" PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};ApparentHorizons" PARENT_SCOPE)
