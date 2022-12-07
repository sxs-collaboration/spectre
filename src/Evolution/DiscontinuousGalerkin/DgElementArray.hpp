// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/DiagnosticInfo.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/OptionTags.hpp"
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
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

namespace detail {
CREATE_HAS_STATIC_MEMBER_VARIABLE(use_z_order_distribution)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(use_z_order_distribution)
}  // namespace detail

/*!
 * \brief The parallel component responsible for managing the DG elements that
 * compose the computational domain
 *
 * This parallel component will perform the actions specified by the
 * `PhaseDepActionList`.
 *
 * The element assignment to processors is performed by
 * `domain::BlockZCurveProcDistribution` (using a Morton space-filling curve),
 * unless `static constexpr bool use_z_order_distribution = false;` is specified
 * in the `Metavariables`, in which case elements are assigned to processors via
 * round-robin assignment. In both cases, an unordered set of `size_t`s can be
 * passed to the `allocate_array` function which represents physical processors
 * to avoid placing elements on.
 */
template <class Metavariables, class PhaseDepActionList>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = ElementId<volume_dim>;

  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<volume_dim>>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_items,
      const std::unordered_set<size_t>& procs_to_ignore = {});

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<DgElementArray>(local_cache)
        .start_phase(next_phase);
  }
};

template <class Metavariables, class PhaseDepActionList>
void DgElementArray<Metavariables, PhaseDepActionList>::allocate_array(
    Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
    const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
        initialization_items,
    const std::unordered_set<size_t>& procs_to_ignore) {
  auto& local_cache = *Parallel::local_branch(global_cache);
  auto& dg_element_array =
      Parallel::get_parallel_component<DgElementArray>(local_cache);

  const auto& domain =
      Parallel::get<domain::Tags::Domain<volume_dim>>(local_cache);
  const auto& initial_refinement_levels =
      get<domain::Tags::InitialRefinementLevels<volume_dim>>(
          initialization_items);
  const auto& initial_extents =
      get<domain::Tags::InitialExtents<volume_dim>>(initialization_items);

  bool use_z_order_distribution = true;
  if constexpr (detail::has_use_z_order_distribution_v<Metavariables>) {
    use_z_order_distribution = Metavariables::use_z_order_distribution;
  }

  const size_t number_of_procs = Parallel::number_of_procs<size_t>(local_cache);
  const size_t number_of_nodes = Parallel::number_of_nodes<size_t>(local_cache);
  const size_t num_of_procs_to_use = number_of_procs - procs_to_ignore.size();

  const domain::BlockZCurveProcDistribution<volume_dim> element_distribution{
      num_of_procs_to_use, initial_refinement_levels, procs_to_ignore};

  // Will be used to print domain diagnostic info
  std::vector<size_t> elements_per_core(number_of_procs, 0_st);
  std::vector<size_t> elements_per_node(number_of_nodes, 0_st);
  std::vector<size_t> grid_points_per_core(number_of_procs, 0_st);
  std::vector<size_t> grid_points_per_node(number_of_nodes, 0_st);

  size_t which_proc = 0;
  for (const auto& block : domain.blocks()) {
    const auto& initial_ref_levs = initial_refinement_levels[block.id()];
    const size_t grid_points_per_element = alg::accumulate(
        initial_extents[block.id()], 1_st, std::multiplies<size_t>());

    const std::vector<ElementId<volume_dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);

    if (use_z_order_distribution) {
      for (const auto& element_id : element_ids) {
        const size_t target_proc =
            element_distribution.get_proc_for_element(element_id);
        dg_element_array(element_id)
            .insert(global_cache, initialization_items, target_proc);

        const size_t target_node =
            Parallel::node_of<size_t>(target_proc, local_cache);
        ++elements_per_core[target_proc];
        ++elements_per_node[target_node];
        grid_points_per_core[target_proc] += grid_points_per_element;
        grid_points_per_node[target_node] += grid_points_per_element;
      }
    } else {
      while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
        which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
      }
      for (size_t i = 0; i < element_ids.size(); ++i) {
        dg_element_array(ElementId<volume_dim>(element_ids[i]))
            .insert(global_cache, initialization_items, which_proc);
        which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;

        const size_t target_node =
            Parallel::node_of<size_t>(which_proc, local_cache);
        ++elements_per_core[which_proc];
        ++elements_per_node[target_node];
        grid_points_per_core[which_proc] += grid_points_per_element;
        grid_points_per_node[target_node] += grid_points_per_element;
      }
    }
  }

  dg_element_array.doneInserting();

  Parallel::printf(
      "\n%s\n", domain::diagnostic_info(domain, local_cache, elements_per_core,
                                        elements_per_node, grid_points_per_core,
                                        grid_points_per_node));
}
