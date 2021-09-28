// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <cstddef>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Protocols/ArrayElementsAllocator.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic {

/*!
 * \brief A `Parallel::protocols::ArrayElementsAllocator` that creates array
 * elements to cover the initial computational domain
 *
 * An element is created for every element ID in every block, determined by the
 * `initial_element_ids` function and the option-created `domain::Tags::Domain`
 * and `domain::Tags::InitialRefinementLevels`. The elements are distributed
 * on processors using the `domain::BlockZCurveProcDistribution`.
 */
template <size_t Dim>
struct DefaultElementsAllocator
    : tt::ConformsTo<Parallel::protocols::ArrayElementsAllocator> {
  template <typename ParallelComponent>
  using array_allocation_tags =
      tmpl::list<domain::Tags::InitialRefinementLevels<Dim>>;

  template <typename ParallelComponent, typename Metavariables,
            typename... InitializationTags>
  static void apply(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::TaggedTuple<InitializationTags...>& initialization_items) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& element_array =
        Parallel::get_parallel_component<ParallelComponent>(local_cache);
    const auto& domain = Parallel::get<domain::Tags::Domain<Dim>>(local_cache);
    const auto& initial_refinement_levels =
        get<domain::Tags::InitialRefinementLevels<Dim>>(initialization_items);
    const domain::BlockZCurveProcDistribution<Dim> element_distribution{
        static_cast<size_t>(sys::number_of_procs()), initial_refinement_levels};
    for (const auto& block : domain.blocks()) {
      const std::vector<ElementId<Dim>> element_ids = initial_element_ids(
          block.id(), initial_refinement_levels[block.id()]);
      for (const auto& element_id : element_ids) {
        const size_t target_proc =
            element_distribution.get_proc_for_element(element_id);
        element_array(element_id)
            .insert(global_cache, initialization_items, target_proc);
      }
    }
    element_array.doneInserting();
  }
};

/*!
 * \brief The parallel component responsible for managing the DG elements that
 * compose the computational domain
 *
 * This parallel component will perform the actions specified by the
 * `PhaseDepActionList`.
 *
 * \note This parallel component is nearly identical to
 * `Evolution/DiscontinuousGalerkin/DgElementArray.hpp` right now, but will
 * likely diverge in the future, for instance to support a multigrid domain.
 *
 */
template <typename Metavariables, typename PhaseDepActionList,
          typename ElementsAllocator =
              DefaultElementsAllocator<Metavariables::volume_dim>>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  static_assert(
      tt::assert_conforms_to<ElementsAllocator,
                             Parallel::protocols::ArrayElementsAllocator>);

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = ElementId<volume_dim>;

  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<volume_dim>>;

  using array_allocation_tags =
      typename ElementsAllocator::template array_allocation_tags<
          DgElementArray>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>,
      array_allocation_tags>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) {
    ElementsAllocator::template apply<DgElementArray>(global_cache,
                                                      initialization_items);
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<DgElementArray>(local_cache)
        .start_phase(next_phase);
  }
};

}  // namespace elliptic
