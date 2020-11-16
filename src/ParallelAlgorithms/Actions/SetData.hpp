// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Mutate the DataBox tags in `TagsList` according to the `data`.
 *
 * An example use case for this action is as the callback for the
 * `importers::ThreadedActions::ReadVolumeData`.
 *
 * DataBox changes:
 * - Modifies:
 *   - All tags in `TagsList`
 */
template <typename TagsList>
struct SetData;

/// \cond
template <typename... Tags>
struct SetData<tmpl::list<Tags...>> {
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      typename ArrayIndex,
      Requires<std::conjunction_v<db::tag_is_retrievable<Tags, DataBox>...>> =
          nullptr>
  static void apply(DataBox& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    tuples::TaggedTuple<Tags...> data) noexcept {
    tmpl::for_each<tmpl::list<Tags...>>([&box, &data](auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      db::mutate<tag>(
          make_not_null(&box), [&data](const gsl::not_null<typename tag::type*>
                                           value) noexcept {
            *value = std::move(tuples::get<tag>(data));
          });
    });
  }
};
/// \endcond

}  // namespace Actions
