// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChangeStepSize.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Time.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Parallel::Tags {
struct Metavariables;
}  // namespace Parallel::Tags
namespace Tags {
struct TimeStep;
struct TimeStepId;
}  // namespace Tags
/// \endcond

/// Bundled method for recording the current system state in the history, and
/// updating the evolved variables and step size.
template <typename System, bool LocalTimeStepping,
          typename StepChoosersToUse = AllStepChoosers, typename DbTags>
void take_step(const gsl::not_null<db::DataBox<DbTags>*> box) {
  if constexpr (LocalTimeStepping) {
    if (db::get<Tags::TimeStepId>(*box).substep() == 0) {
      const auto original_step = db::get<Tags::TimeStep>(*box);
      change_step_size<StepChoosersToUse>(box);
      db::mutate<Tags::AdaptiveSteppingDiagnostics>(
          [&](const gsl::not_null<AdaptiveSteppingDiagnostics*> diags,
              const TimeDelta& new_step) {
            if (original_step != new_step) {
              ++diags->number_of_step_fraction_changes;
            }
          },
          box, db::get<Tags::TimeStep>(*box));
    }
  }
  record_time_stepper_data<System>(box);
  update_u<System>(box);
}
