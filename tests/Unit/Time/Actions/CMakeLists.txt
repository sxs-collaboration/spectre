# Distributed under the MIT License.
# See LICENSE.txt for details.

set(TIME_ACTIONS_TESTS
    Time/Actions/Test_AdvanceTime.cpp
    Time/Actions/Test_FinalTime.cpp
    Time/Actions/Test_UpdateU.cpp
    )

set(TIME_TESTS "${TIME_TESTS};${TIME_ACTIONS_TESTS}" PARENT_SCOPE)
