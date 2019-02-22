// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>
#include <utility>  // IWYU pragma: keep // for std::move

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "Time/Time.hpp" // for Time

/// \cond
// IWYU pragma: no_forward_declare Time
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Records the variables and their time derivatives in the
/// time stepper history.
///
/// With `dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>`:
///
/// Uses:
/// - ConstGlobalCache: nothing
/// - DataBox:
///   - variables_tag
///   - dt_variables_tag
///   - Tags::HistoryEvolvedVariables<system::variables_tag, dt_variables_tag>
///   - Tags::Time
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - dt_variables_tag,
///   - Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>
struct RecordTimeStepperData {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using variables_tag = typename Metavariables::system::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;
    using history_tag =
        Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>;

    db::mutate<dt_variables_tag, history_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<dt_variables_tag>*> dt_vars,
           const gsl::not_null<db::item_type<history_tag>*> history,
           const db::item_type<variables_tag>& vars,
           const db::item_type<Tags::Time>& time) noexcept {
          history->insert(time, vars, std::move(*dt_vars));
        },
        db::get<variables_tag>(box), db::get<Tags::Time>(box));

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
