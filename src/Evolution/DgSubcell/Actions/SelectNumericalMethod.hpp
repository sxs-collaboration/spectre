// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/DgSubcell/Actions/Labels.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Parallel/Actions/Goto.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::subcell::Actions {
/*!
 * \brief Goes to `Labels::BeginDg` or `Labels::BeginSubcell` depending on
 * whether the active grid is `Dg` or `Subcell`.
 *
 * GlobalCache: nothing
 *
 * DataBox:
 * - Uses:
 *   - `subcell::Tags::ActiveGrid`
 */
struct SelectNumericalMethod {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool, size_t> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Note: we jump to the `Label+1` because the label actions don't do
    // anything anyway
    if (db::get<Tags::ActiveGrid>(box) == subcell::ActiveGrid::Dg) {
      const size_t dg_index =
          tmpl::index_of<ActionList, ::Actions::Label<Labels::BeginDg>>::value +
          1;
      return {std::move(box), false, dg_index};
    } else if (db::get<Tags::ActiveGrid>(box) == subcell::ActiveGrid::Subcell) {
      const size_t subcell_index =
          tmpl::index_of<ActionList,
                         ::Actions::Label<Labels::BeginSubcell>>::value +
          1;
      return {std::move(box), false, subcell_index};
    }
    ERROR(
        "Only know DG and subcell active grids for selecting the numerical "
        "method.");
  }
};
}  // namespace evolution::dg::subcell::Actions
