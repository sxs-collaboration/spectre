// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/StepperErrorTolerances.hpp"

/// \ingroup TimeGroup
/// Marker base class for things requiring time stepper error estimates
class RequestsAnyStepperErrorTolerances {
 protected:
  RequestsAnyStepperErrorTolerances() = default;
  ~RequestsAnyStepperErrorTolerances() = default;
};

/// \ingroup TimeGroup
/// Base class for requesting time stepper error tolerances.
template <typename EvolvedVariableTag>
struct RequestsStepperErrorTolerances : RequestsAnyStepperErrorTolerances {
 public:
  virtual StepperErrorTolerances tolerances() const = 0;

 protected:
  RequestsStepperErrorTolerances() = default;
  ~RequestsStepperErrorTolerances() = default;
};
