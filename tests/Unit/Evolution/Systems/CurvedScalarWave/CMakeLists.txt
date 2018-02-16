# Distributed under the MIT License.
# See LICENSE.txt for details.

set(CURVED_SCALAR_WAVE_TESTS
    Evolution/Systems/CurvedScalarWave/TestHelpers.cpp
    Evolution/Systems/CurvedScalarWave/Test_DuDt.cpp
    )

set(SYSTEM_TESTS "${SYSTEM_TESTS};${CURVED_SCALAR_WAVE_TESTS}" PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};CurvedScalarWave" PARENT_SCOPE)
