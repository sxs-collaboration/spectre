# Distributed under the MIT License.
# See LICENSE.txt for details.

set(GENERALIZED_HARMONIC_TESTS
    Evolution/Systems/GeneralizedHarmonic/Test_DuDt.cpp
    )

set(SYSTEM_TESTS "${SYSTEM_TESTS};${GENERALIZED_HARMONIC_TESTS}" PARENT_SCOPE)
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};GeneralizedHarmonic" PARENT_SCOPE)
