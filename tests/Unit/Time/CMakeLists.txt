# Distributed under the MIT License.
# See LICENSE.txt for details.

set(TIME_TESTS
    Time/Test_Slab.cpp
    Time/Test_Time.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${TIME_TESTS}" PARENT_SCOPE)

