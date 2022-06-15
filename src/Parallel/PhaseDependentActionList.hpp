// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Parallel/Phase.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief List of all the actions to be executed in the specified phase.
 */
template <Parallel::Phase Phase, typename ActionsList>
struct PhaseActions {
  using action_list = tmpl::flatten<ActionsList>;
  static constexpr Parallel::Phase phase = Phase;
  static constexpr size_t number_of_actions = tmpl::size<action_list>::value;
};

/*!
 * \ingroup ParallelGroup
 * \brief (Lazy) metafunction to get the action list from a `PhaseActions`
 */
template <typename PhaseDepActionList>
struct get_action_list_from_phase_dep_action_list {
  using type = typename PhaseDepActionList::action_list;
};
}  // namespace Parallel
