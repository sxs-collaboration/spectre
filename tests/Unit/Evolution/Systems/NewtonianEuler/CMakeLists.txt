# Distributed under the MIT License.
# See LICENSE.txt for details.

set(NEWTONIAN_EULER_TESTS
    Evolution/Systems/NewtonianEuler/Test_PrimitiveFromConservative.cpp
    )

set(SYSTEM_TESTS "${SYSTEM_TESTS};${NEWTONIAN_EULER_TESTS}" PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};NewtonianEuler" PARENT_SCOPE)
