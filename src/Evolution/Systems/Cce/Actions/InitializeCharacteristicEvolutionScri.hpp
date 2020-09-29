// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/ScriPlusInterpolationManager.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Initializes the `CharacteristicEvolution` component with contents
 * needed to perform the interpolation at scri+.
 *
 * \details Sets up the \ref DataBoxGroup to be ready to store data in the scri+
 * interpolators and perform interpolation for the final scri+ outputs.
 *
 * \ref DataBoxGroup changes:
 * - Modifies: nothing
 * - Adds:
 *  - `Cce::Tags::InterpolationManager<ComplexDataVector, Tag>` for each `Tag`
 * in `scri_values_to_observe`
 * - Removes: nothing
 */
struct InitializeCharacteristicEvolutionScri {
  using initialization_tags =
      tmpl::list<InitializationTags::ScriInterpolationOrder>;
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;

  template <
      typename DbTags, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<
          DbTags, InitializationTags::ScriInterpolationOrder>> = nullptr>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(initialize_impl(
        std::move(box), typename Metavariables::scri_values_to_observe{}));
  }

  template <typename TagList, typename... TagPack>
  static auto initialize_impl(db::DataBox<TagList>&& box,
                              tmpl::list<TagPack...> /*meta*/) noexcept {
    const size_t target_number_of_points =
        db::get<InitializationTags::ScriInterpolationOrder>(box);
    const size_t vector_size =
        Spectral::Swsh::number_of_swsh_collocation_points(
            db::get<Spectral::Swsh::Tags::LMaxBase>(box));
    // silence compiler warnings when pack is empty
    (void)vector_size;
    return Initialization::merge_into_databox<
        InitializeCharacteristicEvolutionScri,
        db::AddSimpleTags<
            Tags::InterpolationManager<ComplexDataVector, TagPack>...>,
        db::AddComputeTags<>, Initialization::MergePolicy::Overwrite>(
        std::move(box),
        ScriPlusInterpolationManager<ComplexDataVector, TagPack>{
            target_number_of_points, vector_size,
            std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                2 * target_number_of_points - 1,
                2 * target_number_of_points + 2)}...);
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTags, InitializationTags::ScriInterpolationOrder>> = nullptr>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      const db::DataBox<DbTags>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    ERROR(
        "The DataBox is missing required dependency "
        "`Cce::InitializationTags::ScriPlusInterpolationOrder.`");
  }
};
}  // namespace Actions
}  // namespace Cce
