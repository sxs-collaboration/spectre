// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Time.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Parallel::Tags {
struct Metavariables;
}  // namespace Parallel::Tags
/// \endcond

/// Bundled method for recording the current system state in the history, and
/// updating the evolved variables and step size.
///
/// This function is used to encapsulate any needed logic for updating the
/// system, and in the case for which step parameters may need to be rejected
/// and re-tried, looping until an acceptable step is performed.
template <typename System, bool LocalTimeStepping,
          typename StepChoosersToUse = AllStepChoosers, typename DbTags>
void take_step(const gsl::not_null<db::DataBox<DbTags>*> box) {
  record_time_stepper_data<System>(box);
  if constexpr (LocalTimeStepping) {
    uint64_t step_attempts = 0;
    const auto original_step = db::get<Tags::TimeStep>(*box);
    do {
      ++step_attempts;
      update_u<System>(box);
    } while (not change_step_size<StepChoosersToUse>(box));
    db::mutate<Tags::AdaptiveSteppingDiagnostics>(
        [&](const gsl::not_null<AdaptiveSteppingDiagnostics*> diags,
            const TimeDelta& new_step) {
          diags->number_of_step_rejections += step_attempts - 1;
          if (original_step != new_step) {
            ++diags->number_of_step_fraction_changes;
          }
        },
        box, db::get<Tags::TimeStep>(*box));
  } else {
    update_u<System>(box);
  }
}
