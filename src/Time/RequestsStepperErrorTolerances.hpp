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
  ///
  /// \note The use of `std::type_index` is stable in this situation
  /// because while `name()`, `before()`, and `hash()` are
  /// implementation defined or unspecified, the C++ standard
  /// guarantees that two `std::type_index` are equal only if the
  /// underlying types are equal and they are not equal if the types
  /// are different. This means we can use this safely in a
  /// `std::unordered_map`.
  virtual std::unordered_map<std::type_index, StepperErrorTolerances>
  tolerances() const = 0;

 protected:
  RequestsStepperErrorTolerances() = default;
  ~RequestsStepperErrorTolerances() = default;
};
