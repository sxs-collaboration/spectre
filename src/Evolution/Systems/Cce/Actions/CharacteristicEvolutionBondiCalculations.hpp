// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/GaugeTransformBoundaryData.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionGroup
 * \brief Prepare the input quantities in the \ref DataBoxGroup for the
 * evaluation of the hypersurface integral used to compute `BondiTag`.
 *
 * \details Internally this calls the
 * `mutate_all_pre_swsh_derivatives_for_tag<BondiTag>()` and
 * `mutate_all_swsh_derivatives_for_tag<BondiTag>()` utility functions, which
 * determine which quantities are necessary for each of the hypersurface
 * computations.
 */
template <typename BondiTag>
struct CalculateIntegrandInputsForTag {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    mutate_all_pre_swsh_derivatives_for_tag<BondiTag>(make_not_null(&box));
    mutate_all_swsh_derivatives_for_tag<BondiTag>(make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};

/*!
 * \ingroup ActionGroup
 * \brief Perform all of the computations for dependencies of the hypersurface
 * equations that do not themselves depend on any hypersurface integrations.
 *
 * \details This is to be called as a 'pre-computation' step prior to looping
 * over the hypersurface integration set. Internally, this uses
 * `Cce::GaugeAdjustedBoundaryValue<Tag>` and
 * `Cce::mutate_all_precompute_cce_dependencies
 * <Tags::EvolutionGaugeBoundaryValue>()`
 * to calculate the requisite dependencies.
 */
struct PrecomputeGlobalCceDependencies {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    tmpl::for_each<gauge_adjustments_setup_tags>([&box](auto tag_v) noexcept {
      using tag = typename decltype(tag_v)::type;
      db::mutate_apply<GaugeAdjustedBoundaryValue<tag>>(make_not_null(&box));
    });
    mutate_all_precompute_cce_dependencies<Tags::EvolutionGaugeBoundaryValue>(
        make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace Cce
