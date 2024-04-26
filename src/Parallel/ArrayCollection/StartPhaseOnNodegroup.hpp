// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayCollection/ReceiveDataForElement.hpp"
#include "Parallel/ArrayCollection/Tags/ElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel::Actions {
/// \brief Starts the next phase on the nodegroup and calls
/// `ReceiveDataForElement` for each element on the node.
struct StartPhaseOnNodegroup {
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const size_t my_node = Parallel::my_node<size_t>(cache);
    auto proxy_to_this_node =
        Parallel::get_parallel_component<ParallelComponent>(cache)[my_node];
    for (const auto& [element_id, element] :
         db::get<typename ParallelComponent::element_collection_tag>(box)) {
      Parallel::threaded_action<ReceiveDataForElement<true>>(proxy_to_this_node,
                                                             element_id);
    }
    return {Parallel::AlgorithmExecution::Halt, std::nullopt};
  }
};
}  // namespace Parallel::Actions
