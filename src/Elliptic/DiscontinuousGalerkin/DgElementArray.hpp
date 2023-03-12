// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/DiagnosticInfo.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Protocols/ArrayElementsAllocator.hpp"
#include "Utilities/Literals.hpp"
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
 * on processors using the `domain::BlockZCurveProcDistribution`. In both cases,
 * an unordered set of `size_t`s can be passed to the `allocate_array` function
 * which represents physical processors to avoid placing elements on. `Element`s
 * are distributed to processors according to their computational costs
 * determined by the number of grid points.
 */
template <size_t Dim>
struct DefaultElementsAllocator
    : tt::ConformsTo<Parallel::protocols::ArrayElementsAllocator> {
  template <typename ParallelComponent>
  using array_allocation_tags = tmpl::list<>;

  template <typename ParallelComponent, typename Metavariables,
            typename... InitializationTags>
  static void apply(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::TaggedTuple<InitializationTags...>& initialization_items,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& element_array =
        Parallel::get_parallel_component<ParallelComponent>(local_cache);
    const auto& initial_extents =
        get<domain::Tags::InitialExtents<Dim>>(initialization_items);

    const auto& domain = Parallel::get<domain::Tags::Domain<Dim>>(local_cache);
    const auto& initial_refinement_levels =
        get<domain::Tags::InitialRefinementLevels<Dim>>(initialization_items);

    const size_t number_of_procs =
        Parallel::number_of_procs<size_t>(local_cache);
    const size_t number_of_nodes =
        Parallel::number_of_nodes<size_t>(local_cache);
    const size_t num_of_procs_to_use = number_of_procs - procs_to_ignore.size();

    const auto& blocks = domain.blocks();

    const std::unordered_map<ElementId<Dim>, double> element_costs =
        domain::get_element_costs(
            blocks, initial_refinement_levels, initial_extents,
            domain::ElementWeight::NumGridPoints, std::nullopt);
    const domain::BlockZCurveProcDistribution<Dim> element_distribution{
        element_costs,   num_of_procs_to_use, blocks, initial_refinement_levels,
        initial_extents, procs_to_ignore};

    // Will be used to print domain diagnostic info
    std::vector<size_t> elements_per_core(number_of_procs, 0_st);
    std::vector<size_t> elements_per_node(number_of_nodes, 0_st);
    std::vector<size_t> grid_points_per_core(number_of_procs, 0_st);
    std::vector<size_t> grid_points_per_node(number_of_nodes, 0_st);

    for (const auto& block : blocks) {
      const size_t grid_points_per_element = alg::accumulate(
          initial_extents[block.id()], 1_st, std::multiplies<size_t>());

      const std::vector<ElementId<Dim>> element_ids = initial_element_ids(
          block.id(), initial_refinement_levels[block.id()]);

      for (const auto& element_id : element_ids) {
        const size_t target_proc =
            element_distribution.get_proc_for_element(element_id);
        element_array(element_id)
            .insert(global_cache, initialization_items, target_proc);

        const size_t target_node =
            Parallel::node_of<size_t>(target_proc, local_cache);
        ++elements_per_core[target_proc];
        ++elements_per_node[target_node];
        grid_points_per_core[target_proc] += grid_points_per_element;
        grid_points_per_node[target_node] += grid_points_per_element;
      }
    }
    element_array.doneInserting();

    Parallel::printf(
        "\n%s\n", domain::diagnostic_info(
                      domain, local_cache, elements_per_core, elements_per_node,
                      grid_points_per_core, grid_points_per_node));
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
      tt::assert_conforms_to_v<ElementsAllocator,
                               Parallel::protocols::ArrayElementsAllocator>);

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = ElementId<volume_dim>;

  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<volume_dim>>;

  using array_allocation_tags =
      typename ElementsAllocator::template array_allocation_tags<
          DgElementArray>;

  using simple_tags_from_options =
      tmpl::append<Parallel::get_simple_tags_from_options<
                       Parallel::get_initialization_actions_list<
                           phase_dependent_action_list>>,
                   array_allocation_tags>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_items,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    ElementsAllocator::template apply<DgElementArray>(
        global_cache, initialization_items, procs_to_ignore);
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<DgElementArray>(local_cache)
        .start_phase(next_phase);
  }
};

}  // namespace elliptic
