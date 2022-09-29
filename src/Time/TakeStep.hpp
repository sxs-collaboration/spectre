// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Tags.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Utilities/Gsl.hpp"

/// Bundled method for recording the current system state in the history, and
/// updating the evolved variables and step size.
///
/// This function is used to encapsulate any needed logic for updating the
/// system, and in the case for which step parameters may need to be rejected
/// and re-tried, looping until an acceptable step is performed.
template <typename StepChoosersToUse = AllStepChoosers,
          typename VariablesTag = NoSuchType, typename DbTags,
          typename Metavariables>
void take_step(const gsl::not_null<db::DataBox<DbTags>*> box,
               const Parallel::GlobalCache<Metavariables>& cache) {
  using system = typename Metavariables::system;
  record_time_stepper_data<system, VariablesTag>(box);
  if constexpr (Metavariables::local_time_stepping) {
    for (;;) {
      update_u<system, VariablesTag>(box);
      if (change_step_size<StepChoosersToUse>(box, cache)) {
        break;
      }
      using variables_tag =
          tmpl::conditional_t<std::is_same_v<VariablesTag, NoSuchType>,
                              typename system::variables_tag, VariablesTag>;
      using rollback_tag = Tags::RollbackValue<variables_tag>;
      db::mutate<variables_tag>(
          box,
          [](const gsl::not_null<typename variables_tag::type*> vars,
             const typename rollback_tag::type& rollback_value) {
            *vars = rollback_value;
          },
          db::get<rollback_tag>(*box));
    }
  } else {
    update_u<system, VariablesTag>(box);
  }
}
