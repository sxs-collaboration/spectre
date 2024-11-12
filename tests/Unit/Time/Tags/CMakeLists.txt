# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY_SOURCES
  ${LIBRARY_SOURCES}
  Tags/Test_AdaptiveSteppingDiagnostics.cpp
  Tags/Test_FixedLtsRatio.cpp
  Tags/Test_HistoryEvolvedVariables.cpp
  Tags/Test_IsUsingTimeSteppingErrorControl.cpp
  Tags/Test_MinimumTimeStep.cpp
  Tags/Test_StepChoosers.cpp
  Tags/Test_StepNumberWithinSlab.cpp
  Tags/Test_StepperErrorTolerances.cpp
  Tags/Test_StepperErrors.cpp
  Tags/Test_Time.cpp
  Tags/Test_TimeAndPrevious.cpp
  Tags/Test_TimeStepId.cpp
  Tags/Test_TimeStepper.cpp
  PARENT_SCOPE)
