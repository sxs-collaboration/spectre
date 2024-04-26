// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayCollection/ReceiveDataForElement.hpp"
#include "Parallel/ArrayCollection/Tags/ElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"

namespace Parallel::Actions {
/*!
 * \brief Reduction target that is called after all nodes have successfully
 * initialized the nodegroup portion of the `DgElementCollection`. This spawns
 * the messages that initialize the `DgElementArrayMember`s
 */
struct SpawnInitializeElementsInCollection {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const double /*unused_but_we_needed_to_reduce_something*/) {
    auto my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    db::mutate<typename ParallelComponent::element_collection_tag>(
        [&my_proxy](const auto element_collection_ptr) {
          for (auto& [element_id, element] : *element_collection_ptr) {
            Parallel::threaded_action<ReceiveDataForElement<true>>(my_proxy,
                                                                   element_id);
          }
        },
        make_not_null(&box));
  }
};
}  // namespace Parallel::Actions
