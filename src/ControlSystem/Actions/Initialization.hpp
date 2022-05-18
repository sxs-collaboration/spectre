// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>
#include <utility>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/Tags.hpp"
#include "ControlSystem/Tags/MeasurementTimescales.hpp"
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
 * GlobalCache:
 * - Uses:
 *   - `control_system::Tags::MeasurementTimescales`
 *
 * DataBox:
 * - Uses: Nothing
 * - Adds:
 *   - `control_system::Tags::Averager<ContolSystem>`
 *   - `control_system::Tags::Controller<ControlSystem>`
 *   - `control_system::Tags::TimescaleTuner<ControlSystem>`
 *   - `control_system::Tags::ControlError<ControlSystem>`
 *   - `control_system::Tags::WriteDataToDisk`
 *   - `control_system::Tags::IsActive<ControlSystem>`
 * - Removes: Nothing
 * - Modifies:
 *   - `control_system::Tags::Averager<ControlSystem>`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename Metavariables, typename ControlSystem>
struct Initialize {
  static constexpr size_t deriv_order = ControlSystem::deriv_order;

  using initialization_tags =
      tmpl::list<control_system::Tags::WriteDataToDisk,
                 control_system::Tags::Averager<ControlSystem>,
                 control_system::Tags::Controller<ControlSystem>,
                 control_system::Tags::TimescaleTuner<ControlSystem>,
                 control_system::Tags::ControlError<ControlSystem>,
                 control_system::Tags::IsActive<ControlSystem>>;

  using initialization_tags_to_keep = initialization_tags;

  using simple_tags = typename ControlSystem::simple_tags;

  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    // Set the initial time between updates and measurements
    const auto& measurement_timescales =
        get<control_system::Tags::MeasurementTimescales>(cache);
    const auto& measurement_timescale_func =
        *(measurement_timescales.at(ControlSystem::name()));
    const double initial_time = measurement_timescale_func.time_bounds()[0];
    const double measurement_timescale =
        min(measurement_timescale_func.func(initial_time)[0]);
    db::mutate<control_system::Tags::Averager<ControlSystem>>(
        make_not_null(&box),
        [&measurement_timescale](
            const gsl::not_null<::Averager<deriv_order - 1>*> averager) {
          averager->assign_time_between_measurements(measurement_timescale);
        });

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace control_system
