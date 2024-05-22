// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags/ElementDistribution.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayCollection/SpawnInitializeElementsInCollection.hpp"
#include "Parallel/ArrayCollection/Tags/ElementCollection.hpp"
#include "Parallel/ArrayCollection/Tags/ElementLocations.hpp"
#include "Parallel/ArrayCollection/Tags/NumberOfElementsTerminated.hpp"
#include "Parallel/CreateElementsUsingDistribution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel::Actions {
/*!
 * \brief Creates the `DgElementArrayMember`s on the (node)group component.
 *
 * First the distribution of elements is computed using
 * `Parallel::create_elements_using_distribution()`, they are inserted on
 * each (node)group element. A reduction is done over the (node)group before
 * initializing the `DgElementArrayMember`s themselves, since they are allowed
 * to communicate with each other during their initialization. The reduction
 * target is `Parallel::Actions::SpawnInitializeElementsInCollection`.
 *
 * Uses:
 * - DataBox:
 *   - `domain::Tags::Domain<Dim>`
 *   - `domain::Tags::InitialRefinementLevels<Dim>`
 *   - `domain::Tags::InitialExtents<Dim>`
 *   - `evolution::dg::Tags::Quadrature`
 *   - `domain::Tags::ElementDistribution`
 *
 * DataBox changes:
 * - Adds:
 *   - `Parallel::Tags::ElementCollection`
 *   - `Parallel::Tags::ElementLocations<Dim>`
 *   - `Parallel::Tags::NumberOfElementsTerminated`
 * - Removes: nothing
 * - Modifies:
 *   - `Parallel::Tags::ElementCollection`
 *   - `Parallel::Tags::ElementLocations<Dim>`
 *   - `Parallel::Tags::NumberOfElementsTerminated`
 */
template <size_t Dim, class Metavariables, class PhaseDepActionList,
          typename SimpleTagsFromOptions>
struct CreateElementCollection {
  using simple_tags = tmpl::list<
      Parallel::Tags::ElementCollection<Dim, Metavariables, PhaseDepActionList,
                                        SimpleTagsFromOptions>,
      Parallel::Tags::ElementLocations<Dim>, Tags::NumberOfElementsTerminated>;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<::domain::Tags::Domain<Dim>,
                 ::domain::Tags::ElementDistribution>;

  using return_tag_list = tmpl::append<simple_tags, compute_tags>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& local_cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const std::unordered_set<size_t> procs_to_ignore{};

    const auto& domain = Parallel::get<domain::Tags::Domain<Dim>>(local_cache);
    const auto& initial_refinement_levels =
        get<domain::Tags::InitialRefinementLevels<Dim>>(box);
    const auto& initial_extents = get<domain::Tags::InitialExtents<Dim>>(box);
    const auto& quadrature = get<evolution::dg::Tags::Quadrature>(box);
    const std::optional<domain::ElementWeight>& element_weight =
        Parallel::get<domain::Tags::ElementDistribution>(local_cache);

    const size_t number_of_procs =
        Parallel::number_of_procs<size_t>(local_cache);
    const size_t number_of_nodes =
        Parallel::number_of_nodes<size_t>(local_cache);
    const size_t num_of_procs_to_use = number_of_procs - procs_to_ignore.size();

    const auto& blocks = domain.blocks();

    const size_t total_num_elements = [&blocks, &initial_refinement_levels]() {
      size_t result = 0;
      for (const auto& block : blocks) {
        const auto& initial_ref_levs = initial_refinement_levels[block.id()];
        for (const size_t ref_lev : initial_ref_levs) {
          result += two_to_the(ref_lev);
        }
      }
      return result;
    }();
    std::vector<std::pair<ElementId<Dim>, size_t>> my_elements_and_cores{};
    my_elements_and_cores.reserve(total_num_elements / number_of_nodes + 1);
    std::unordered_map<ElementId<Dim>, size_t> node_of_elements{};
    const size_t my_node = Parallel::my_node<size_t>(local_cache);

    Parallel::create_elements_using_distribution(
        [&my_elements_and_cores, my_node, &node_of_elements](
            const ElementId<Dim>& element_id, const size_t target_proc,
            const size_t target_node) {
          node_of_elements.insert(std::pair{element_id, target_node});
          if (target_node == my_node) {
            my_elements_and_cores.push_back(std::pair{element_id, target_proc});
          }
        },
        element_weight, blocks, initial_extents, initial_refinement_levels,
        quadrature,
        // The below arguments control how the elements are mapped to the
        // hardware.
        procs_to_ignore, number_of_procs, number_of_nodes, num_of_procs_to_use,
        local_cache, my_node == 0);

    tuples::tagged_tuple_from_typelist<SimpleTagsFromOptions>
        initialization_items = db::copy_items<SimpleTagsFromOptions>(box);

    const gsl::not_null<Parallel::NodeLock*> node_lock = make_not_null(
        &Parallel::local_branch(
             Parallel::get_parallel_component<ParallelComponent>(local_cache))
             ->get_node_lock());
    db::mutate<Tags::ElementLocations<Dim>,
               Tags::ElementCollection<Dim, Metavariables, PhaseDepActionList,
                                       SimpleTagsFromOptions>,
               Tags::NumberOfElementsTerminated>(
        [&local_cache, &initialization_items, &my_elements_and_cores,
         &node_of_elements](
            const auto element_locations_ptr, const auto collection_ptr,
            const gsl::not_null<size_t*> number_of_elements_terminated) {
          *number_of_elements_terminated = 0;
          const auto serialized_initialization_items =
              serialize(initialization_items);
          *element_locations_ptr = std::move(node_of_elements);
          for (const auto& [element_id, core] : my_elements_and_cores) {
            if (not collection_ptr
                        ->emplace(
                            std::piecewise_construct,
                            std::forward_as_tuple(element_id),
                            std::forward_as_tuple(
                                local_cache.get_this_proxy(),
                                deserialize<tuples::tagged_tuple_from_typelist<
                                    SimpleTagsFromOptions>>(
                                    serialized_initialization_items.data()),
                                element_id))
                        .second) {
              ERROR("Failed to insert element with ID: " << element_id);
            }
            if (collection_ptr->at(element_id).get_terminate() == true) {
              ++(*number_of_elements_terminated);
            } else {
              ERROR("Inserted element with ID "
                    << element_id
                    << " was not initialized in a terminated state. This is a "
                       "bug.");
            }
            collection_ptr->at(element_id).set_core(core);
          }
          ASSERT(*number_of_elements_terminated == collection_ptr->size(),
                 "The number of elements inserted must match the number of "
                 "elements set to terminate since the default state of an "
                 "inserted element is terminated. This is a bug.");
        },
        make_not_null(&box));

    Parallel::contribute_to_reduction<
        Parallel::Actions::SpawnInitializeElementsInCollection>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<int, funcl::AssertEqual<>>>{0},
        Parallel::get_parallel_component<ParallelComponent>(
            local_cache)[my_node],
        Parallel::get_parallel_component<ParallelComponent>(local_cache));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Parallel::Actions
