// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Utilities/Gsl.hpp"

/// Bundled method for recording the current system state in the history, and
/// updating the evolved variables and step size.
///
/// This function is used to encapsulate any needed logic for updating the
/// system, and in the case for which step parameters may need to be rejected
/// and re-tried, looping until an acceptable step is performed.
template <typename VariablesTag = NoSuchType, typename DbTags,
          typename Metavariables>
void take_step(const gsl::not_null<db::DataBox<DbTags>*> box,
               const Parallel::GlobalCache<Metavariables>& cache) {
  record_time_stepper_data<typename Metavariables::system, VariablesTag>(box);
  do {
    update_u<typename Metavariables::system, VariablesTag>(box);
  } while (Metavariables::local_time_stepping and
           not change_step_size(box, cache));
}
