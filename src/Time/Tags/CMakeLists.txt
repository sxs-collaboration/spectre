# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdaptiveSteppingDiagnostics.hpp
  HistoryEvolvedVariables.hpp
  IsUsingTimeSteppingErrorControl.hpp
  StepChoosers.hpp
  StepperErrorTolerances.hpp
  StepperErrors.hpp
  Time.hpp
  TimeAndPrevious.hpp
  TimeStep.hpp
  TimeStepId.hpp
  TimeStepper.hpp
  )
