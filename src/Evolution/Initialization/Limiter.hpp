// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/SizeOfElement.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
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
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        merge_into_databox<Minmod, db::AddSimpleTags<>, compute_tags>(
            std::move(box)));
  }
};
}  // namespace Actions
}  // namespace Initialization
