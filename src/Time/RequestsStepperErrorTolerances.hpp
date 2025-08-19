// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <typeindex>
#include <unordered_map>

#include "Time/StepperErrorTolerances.hpp"

/// \ingroup TimeGroup
/// Base class for requesting time stepper error tolerances.
struct RequestsStepperErrorTolerances {
 public:
  /// A map from the type of a variables tag to the tolerances for
  /// that variable.
  virtual std::unordered_map<std::type_index, StepperErrorTolerances>
  tolerances() const = 0;

 protected:
  RequestsStepperErrorTolerances() = default;
  ~RequestsStepperErrorTolerances() = default;
};
