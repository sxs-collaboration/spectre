// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Initialization {
namespace Actions {
/// \ingroup InitializationGroup
/// \brief Allocate items for minmod limiter
///
/// DataBox changes:
/// - Adds:
///   * `Tags::SizeOfElement<Dim>`
///
/// - Removes: nothing
/// - Modifies: nothing
template <size_t Dim>
struct Minmod {
  using simple_tags = db::AddSimpleTags<>;
  using compute_tags = tmpl::list<domain::Tags::SizeOfElementCompute<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Initialization
