# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  StepperErrorTolerancesCompute.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdaptiveSteppingDiagnostics.hpp
  FixedLtsRatio.hpp
  HistoryEvolvedVariables.hpp
  MinimumTimeStep.hpp
  StepChoosers.hpp
  StepNumberWithinSlab.hpp
  StepperErrorEstimatesEnabled.hpp
  StepperErrorTolerances.hpp
  StepperErrorTolerancesCompute.hpp
  StepperErrors.hpp
  Time.hpp
  TimeAndPrevious.hpp
  TimeStep.hpp
  TimeStepId.hpp
  TimeStepper.hpp
  VariableOrderAlgorithm.hpp
  )
