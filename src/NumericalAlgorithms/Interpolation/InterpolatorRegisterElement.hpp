// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "ParallelBackend/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Called by each local Element to register itself with an Interpolator.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::NumberOfElements`
///
/// For requirements on Metavariables, see InterpolationTarget.
struct RegisterElement {
  template <
      typename ParallelComponent, typename DbTags, typename Metavariables,
      typename ArrayIndex,
      Requires<tmpl::list_contains_v<DbTags, Tags::NumberOfElements>> = nullptr>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/) noexcept {
    db::mutate<Tags::NumberOfElements>(
        make_not_null(&box), [](const gsl::not_null<
                                 db::item_type<Tags::NumberOfElements>*>
                                    num_elements) noexcept {
          ++(*num_elements);
        });
  }
};

}  // namespace Actions
}  // namespace intrp
