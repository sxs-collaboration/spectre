# Distributed under the MIT License.
# See LICENSE.txt for details.

set(CONTROL_SYSTEM_TESTS
    ControlSystem/Test_PiecewisePolynomial.cpp
    ControlSystem/Test_SettleToConstant.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${CONTROL_SYSTEM_TESTS}" PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};ControlSystem" PARENT_SCOPE)
