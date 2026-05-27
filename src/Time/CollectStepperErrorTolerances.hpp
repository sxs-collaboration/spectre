// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <typeinfo>
#include <unordered_map>

#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
struct StepperErrorTolerances;
/// \endcond

namespace collect_stepper_error_tolerances_detail {
void process_request(
    gsl::not_null<std::unordered_map<std::type_index, StepperErrorTolerances>*>
        tolerances,
    const RequestsStepperErrorTolerances& tolerance_request);
}  // namespace collect_stepper_error_tolerances_detail

/// If \p potential_request inherits from
/// `RequestsStepperErrorTolerances`, merge its tolerance requests
/// into \p tolerances.
template <typename PotentialRequest>
void collect_stepper_error_tolerances(
    const gsl::not_null<
        std::unordered_map<std::type_index, StepperErrorTolerances>*>
        tolerances,
    const PotentialRequest& potential_request) {
  // This will fail to compile for non-polymorphic types.  We could
  // check for that, but all normal uses of this should be on
  // polymorphic types and it serves as a check that you don't pass a
  // (smart) pointer to the object instead of the object itself.
  if (const auto* const tolerance_request =
          dynamic_cast<const RequestsStepperErrorTolerances*>(
              &potential_request);
      tolerance_request != nullptr) {
    collect_stepper_error_tolerances_detail::process_request(
        tolerances, *tolerance_request);
  }
}
