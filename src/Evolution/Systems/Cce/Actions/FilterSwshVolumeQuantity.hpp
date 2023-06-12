// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Filters the spherical volume data stored in `BondiTag` according to
 * the filter parameters in the `Parallel::GlobalCache`.
 *
 * \details This action dispatches to the function
 * `filter_swsh_volume_quantity()` to perform the mathematics of
 * the filtering
 *
 * Uses:
 * - DataBox:
 *   - `Cce::Tags::LMax`
 * - GlobalCache:
 *   - `InitializationTags::FilterLMax`
 *   - `InitializationTags::RadialFilterAlpha`
 *   - `InitializationTags::RadialFilterHalfPower`
 * \ref DataBoxGroup changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: `BondiTag`
 */
template <typename BondiTag>
struct FilterSwshVolumeQuantity {
  using const_global_cache_tags =
      tmpl::list<Tags::FilterLMax, Tags::RadialFilterAlpha,
                 Tags::RadialFilterHalfPower>;
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t l_max = db::get<Tags::LMax>(box);
    const size_t l_filter_start = get<Tags::FilterLMax>(box);
    const double radial_filter_alpha = get<Tags::RadialFilterAlpha>(box);
    const size_t radial_filter_half_power =
        get<Tags::RadialFilterHalfPower>(box);
    db::mutate<BondiTag>(
        [&l_max, &l_filter_start, &radial_filter_alpha,
         &radial_filter_half_power](
            const gsl::not_null<typename BondiTag::type*> bondi_quantity) {
          Spectral::Swsh::filter_swsh_volume_quantity(
              make_not_null(&get(*bondi_quantity)), l_max, l_filter_start,
              radial_filter_alpha, radial_filter_half_power);
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace Cce
