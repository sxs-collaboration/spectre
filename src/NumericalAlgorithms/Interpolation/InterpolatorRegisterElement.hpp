// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp" // IWYU pragma: keep
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
template <typename Metavariables>
struct Interpolator;
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Invoked on the `Interpolator` ParallelComponent to register an
/// element with the `Interpolator`.
///
/// This is called by `RegisterElementWithInterpolator` below.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::NumberOfElements`
///
/// For requirements on Metavariables, see `InterpolationTarget`.
struct RegisterElement {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTags, Tags::NumberOfElements>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) noexcept {
    db::mutate<Tags::NumberOfElements>(
        make_not_null(&box), [](const gsl::not_null<
                                 db::item_type<Tags::NumberOfElements>*>
                                    num_elements) noexcept {
          ++(*num_elements);
        });
  }
};

/// \ingroup ActionsGroup
/// \brief Invoked on `DgElementArray` to register all its elements with the
/// `Interpolator`.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
///
struct RegisterElementWithInterpolator {
  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagList>&&> apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto& interpolator =
        *Parallel::get_parallel_component<::intrp::Interpolator<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<RegisterElement>(interpolator);
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace intrp
