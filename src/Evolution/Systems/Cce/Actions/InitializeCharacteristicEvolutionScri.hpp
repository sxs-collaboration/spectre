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
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
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
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename ScriValuesToObserve>
struct InitializeCharacteristicEvolutionScri {
  using initialization_tags =
      tmpl::list<InitializationTags::ScriInterpolationOrder>;
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;

  using simple_tags =
      tmpl::transform<ScriValuesToObserve,
                      tmpl::bind<Tags::InterpolationManager,
                                 tmpl::pin<ComplexDataVector>, tmpl::_1>>;

  using compute_tags = tmpl::list<>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    initialize_impl(make_not_null(&box),
                    typename Metavariables::scri_values_to_observe{});
    return std::make_tuple(std::move(box));
  }

  template <typename TagList, typename... TagPack>
  static void initialize_impl(const gsl::not_null<db::DataBox<TagList>*> box,
                              tmpl::list<TagPack...> /*meta*/) noexcept {
    const size_t target_number_of_points =
        db::get<InitializationTags::ScriInterpolationOrder>(*box);
    const size_t vector_size =
        Spectral::Swsh::number_of_swsh_collocation_points(
            db::get<Spectral::Swsh::Tags::LMaxBase>(*box));
    // silence compiler warnings when pack is empty
    (void)vector_size;
    if constexpr (sizeof...(TagPack) > 0) {
      Initialization::mutate_assign<simple_tags>(
          box, ScriPlusInterpolationManager<ComplexDataVector, TagPack>{
                   target_number_of_points, vector_size,
                   std::make_unique<intrp::BarycentricRationalSpanInterpolator>(
                       2 * target_number_of_points - 1,
                       2 * target_number_of_points + 2)}...);
    }
  }
};
}  // namespace Actions
}  // namespace Cce
