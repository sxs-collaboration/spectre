// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionGroup
 * \brief Filters the spherical volume data stored in `BondiTag` according to
 * the filter parameters in the `Parallel::ConstGlobalCache`.
 *
 * \details This action dispatches to the function
 * `filter_swsh_volume_quantity()` to perform the mathematics of
 * the filtering
 *
 * Uses:
 * - DataBox:
 *   - `Spectral::Swsh::Tags::LMax`
 * - ConstGlobalCache:
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
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const size_t l_max = db::get<Spectral::Swsh::Tags::LMax>(box);
    const size_t l_filter_start = get<Tags::FilterLMax>(box);
    const double radial_filter_alpha = get<Tags::RadialFilterAlpha>(box);
    const size_t radial_filter_half_power =
        get<Tags::RadialFilterHalfPower>(box);
    db::mutate<BondiTag>(
        make_not_null(&box),
        [
          &l_max, &l_filter_start, &radial_filter_alpha, &
          radial_filter_half_power
        ](const gsl::not_null<db::item_type<BondiTag>*>
              bondi_quantity) noexcept {
          Spectral::Swsh::filter_swsh_volume_quantity(
              make_not_null(&get(*bondi_quantity)), l_max, l_filter_start,
              radial_filter_alpha, radial_filter_half_power);
        });
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace Cce
