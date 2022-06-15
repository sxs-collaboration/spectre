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
  using phase_type = Parallel::Phase;
  static constexpr Parallel::Phase phase = Phase;
  using integral_constant_phase =
      std::integral_constant<Parallel::Phase, Phase>;
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

/*!
 * \ingroup ParallelGroup
 * \brief (Lazy) metafunction to get the phase type from a `PhaseActions`
 */
template <typename PhaseDepActionList>
struct get_phase_type_from_phase_dep_action_list {
  using type = Parallel::Phase;
};

/*!
 * \ingroup ParallelGroup
 * \brief (Lazy) metafunction to get the phase as a `std::integral_constant`
 * from a `PhaseActions`
 */
template <typename PhaseDepActionList>
struct get_phase_from_phase_dep_action_list {
  using type = typename PhaseDepActionList::integral_constant_phase;
};
}  // namespace Parallel
