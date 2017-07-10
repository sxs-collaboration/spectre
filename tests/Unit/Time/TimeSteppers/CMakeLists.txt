# Distributed under the MIT License.
# See LICENSE.txt for details.

set(TIME_STEPPER_TESTS
    Time/TimeSteppers/Test_AdamsBashforthN.cpp
    Time/TimeSteppers/Test_RungeKutta3.cpp
   )

set(TIME_TESTS "${TIME_TESTS};${TIME_STEPPER_TESTS}"
    PARENT_SCOPE)
