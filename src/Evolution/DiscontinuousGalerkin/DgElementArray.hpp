// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags/ElementDistribution.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/ArrayCollection/CreateElementsUsingDistribution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \brief The parallel component responsible for managing the DG elements that
 * compose the computational domain
 *
 * This parallel component will perform the actions specified by the
 * `PhaseDepActionList`.
 *
 * An unordered set of `size_t`s can be passed to the `allocate_array`
 * function which represents physical processors to avoid placing
 * elements on.
 */
template <class Metavariables, class PhaseDepActionList>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  static constexpr bool checkpoint_data = true;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = ElementId<volume_dim>;

  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<volume_dim>,
                                             domain::Tags::ElementDistribution>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  using array_allocation_tags = tmpl::list<>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_items,
      const tuples::tagged_tuple_from_typelist<array_allocation_tags>&
          array_allocation_items = {},
      const std::unordered_set<size_t>& procs_to_ignore = {});

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const bool force = false) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<DgElementArray>(local_cache)
        .start_phase(next_phase, force);
  }
};

template <class Metavariables, class PhaseDepActionList>
void DgElementArray<Metavariables, PhaseDepActionList>::allocate_array(
    Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
    const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
        initialization_items,
    const tuples::tagged_tuple_from_typelist<array_allocation_tags>&
    /*array_allocation_items*/,
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
  const auto i1_basis = Spectral::Basis::Legendre;
  const auto& i1_quadrature =
      get<evolution::dg::Tags::Quadrature>(initialization_items);
  const std::optional<domain::ElementWeight>& element_weight =
      Parallel::get<domain::Tags::ElementDistribution>(local_cache);

  const size_t number_of_procs = Parallel::number_of_procs<size_t>(local_cache);
  const size_t number_of_nodes = Parallel::number_of_nodes<size_t>(local_cache);
  const size_t num_of_procs_to_use = number_of_procs - procs_to_ignore.size();

  const auto& blocks = domain.blocks();

  Parallel::create_elements_using_distribution(
      [&dg_element_array, &global_cache, &initialization_items](
          const ElementId<volume_dim>& element_id, const size_t target_proc,
          const size_t /*target_node*/) {
        dg_element_array(element_id)
            .insert(global_cache, initialization_items, target_proc);
      },
      element_weight, blocks, initial_extents, initial_refinement_levels,
      i1_basis, i1_quadrature,

      procs_to_ignore, number_of_procs, number_of_nodes, num_of_procs_to_use,
      local_cache, true);
  dg_element_array.doneInserting();
}
