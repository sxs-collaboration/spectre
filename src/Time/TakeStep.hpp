// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/ChangeStepSize.hpp"
#include "Utilities/Gsl.hpp"

/// Bundled method for recording the current system state in the history, and
/// updating the evolved variables and step size.
template <typename System, bool LocalTimeStepping,
          typename StepChoosersToUse = AllStepChoosers, typename DbTags>
void take_step(const gsl::not_null<db::DataBox<DbTags>*> box) {
  if constexpr (LocalTimeStepping) {
    db::mutate_apply<ChangeStepSize<StepChoosersToUse>>(box);
  }
  record_time_stepper_data<System>(box);
  update_u<System>(box);
}
