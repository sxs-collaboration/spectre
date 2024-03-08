// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"

namespace Cce::Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Compute the set of inputs to `ComputeKleinGordonSource`.
 *
 * \details \ref DataBoxGroup changes:
 * - Modifies:
 *  - `Tags::Dy<Tags::KleinGordonPsi>`
 *  - `Spectral::Swsh::Tags::Derivative<Tags::KleinGordonPsi,
                                                  Spectral::Swsh::Tags::Eth>`
 * - Adds: nothing
 * - Removes: nothing
 */
struct PrecomputeKleinGordonSourceVariables {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    mutate_all_pre_swsh_derivatives_for_tag<
        Tags::KleinGordonSource<Tags::BondiBeta>>(make_not_null(&box));
    mutate_all_swsh_derivatives_for_tag<Tags::KleinGordonSource<Tags::BondiQ>>(
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Cce::Actions
