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
 * - Uses: Nothing
 * - Adds: Nothing
 * - Removes: Nothing
 * - Modifies:
 *   - `control_system::Tags::ControlSystemName`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename Metavariables, typename ControlSystem>
struct Initialize {
  // The averager here and controller below are hard coded with a DerivOrder=2
  // at the moment, because this is the DerivOrder of most functions of time in
  // the domain creators. If we want to choose the DerivOrder at runtime (i.e.
  // in an input file), that capability will need to be added later.
  using initialization_tags = tmpl::list<control_system::Tags::Averager<2>,
                                         control_system::Tags::TimescaleTuner>;

  using initialization_tags_to_keep = initialization_tags;

  using simple_tags = tmpl::push_back<typename ControlSystem::simple_tags,
                                      control_system::Tags::Controller<2>>;

  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) {
    db::mutate<control_system::Tags::ControlSystemName>(
        make_not_null(&box), [](const gsl::not_null<std::string*> tag0) {
          *tag0 = ControlSystem::name();
        });
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace control_system
