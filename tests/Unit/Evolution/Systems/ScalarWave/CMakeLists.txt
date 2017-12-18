# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SCALAR_WAVE_TESTS
    Evolution/Systems/ScalarWave/Test_Equations.cpp
    )

set(SYSTEM_TESTS "${SYSTEM_TESTS};${SCALAR_WAVE_TESTS}" PARENT_SCOPE)
