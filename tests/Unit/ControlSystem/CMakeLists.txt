# Distributed under the MIT License.
# See LICENSE.txt for details.

set(CONTROL_SYSTEM_TESTS
    ControlSystem/Test_ControlSys.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${CONTROL_SYSTEM_TESTS}" PARENT_SCOPE)
