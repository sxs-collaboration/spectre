// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>
#include <utility>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace control_system {
namespace Actions {
/*!
 * \ingroup ControlSystemGroup
 * \ingroup InitializationGroup
 * \brief Initialize items related to the control system
 *
 * DataBox:
 * - Uses:
 *   - `control_system::Tags::ControlSystemInputs<ControlSystem>`
 * - Adds:
 *   - `control_system::Tags::Averager<2>`
 *   - `control_system::Tags::Controller<2>`
 *   - `control_system::Tags::TimescaleTuner`
 *   - `control_system::Tags::ControlSystemName`
 * - Removes: Nothing
 * - Modifies:
 *   - `control_system::Tags::Controller<2>`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename Metavariables, typename ControlSystem>
struct Initialize {
  static constexpr size_t deriv_order = ControlSystem::deriv_order;

  using initialization_tags =
      tmpl::list<control_system::Tags::ControlSystemInputs<ControlSystem>>;

  using tags_to_be_initialized =
      tmpl::list<control_system::Tags::Averager<deriv_order>,
                 control_system::Tags::Controller<deriv_order>,
                 control_system::Tags::TimescaleTuner,
                 control_system::Tags::ControlSystemName>;

  using simple_tags =
      tmpl::append<tags_to_be_initialized, typename ControlSystem::simple_tags>;

  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    // Move all the control system inputs into their own tags in the databox so
    // we can use them easily later
    const auto& option_holder =
        db::get<control_system::Tags::ControlSystemInputs<ControlSystem>>(box);
    ::Initialization::mutate_assign<tags_to_be_initialized>(
        make_not_null(&box), option_holder.averager, option_holder.controller,
        option_holder.tuner, ControlSystem::name());

    // Set the initial time between updates using the initial timescale
    const auto& tuner = db::get<control_system::Tags::TimescaleTuner>(box);
    const double current_min_timescale = min(tuner.current_timescale());
    db::mutate<control_system::Tags::Controller<deriv_order>>(
        make_not_null(&box),
        [&current_min_timescale](
            const gsl::not_null<::Controller<deriv_order>*> controller) {
          controller->assign_time_between_updates(current_min_timescale);
        });

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace control_system
