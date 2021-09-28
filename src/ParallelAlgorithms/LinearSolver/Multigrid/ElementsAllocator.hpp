// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <charm++.h>
#include <cstddef>
#include <optional>
#include <vector>

#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Protocols/ArrayElementsAllocator.hpp"
#include "Parallel/Section.hpp"
#include "Parallel/Serialize.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Hierarchy.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace LinearSolver::multigrid {

/*!
 * \brief A `Parallel::protocols::ArrayElementsAllocator` that creates array
 * elements to cover the initial computational domain multiple times at
 * different refinement levels, suitable for the
 * `LinearSolver::multigrid::Multigrid` algorithm
 *
 * The `initial_element_ids` function and the option-created
 * `domain::Tags::Domain` and `domain::Tags::InitialRefinementLevels` determine
 * the initial set of element IDs in the domain. This is taken as the finest
 * grid in the multigrid hierarchy. Coarser grids are determined by successively
 * applying `LinearSolver::multigrid::coarsen`, up to
 * `LinearSolver::multigrid::Tags::MaxLevels` grids.
 *
 * Array elements are created for all element IDs on all grids, meaning they all
 * share the same parallel component, action list etc. Elements are connected to
 * their neighbors _on the same grid_ by the
 * `domain::Initialization::create_initial_element` function (this is
 * independent of the multigrid code, the function is typically called in an
 * initialization action). Elements are connected to their parent and children
 * _across grids_ by the multigrid-tags set here and in the multigrid
 * initialization actions (see `LinearSolver::multigrid::parent_id` and
 * `LinearSolver::multigrid::child_ids`).
 *
 * This allocator also creates two sets of sections (see `Parallel::Section`):
 * - `Parallel::Tags::Section<ElementArray,
 *   LinearSolver::multigrid::Tags::MultigridLevel>`: One section per grid. All
 *   elements are part of a section with this tag.
 * - `Parallel::Tags::Section<ElementArray,
 *   LinearSolver::multigrid::Tags::IsFinestGrid>`: A single section that holds
 *   only the elements on the finest grid. Holds `std::nullopt` on all other
 *   elements.
 *
 * The elements are distributed on processors using the
 * `domain::BlockZCurveProcDistribution` for every grid independently.
 */
template <size_t Dim, typename OptionsGroup>
struct ElementsAllocator
    : tt::ConformsTo<Parallel::protocols::ArrayElementsAllocator> {
  template <typename ElementArray>
  using array_allocation_tags =
      tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                 Tags::ChildrenRefinementLevels<Dim>,
                 Tags::ParentRefinementLevels<Dim>,
                 Parallel::Tags::Section<ElementArray, Tags::MultigridLevel>,
                 Parallel::Tags::Section<ElementArray, Tags::IsFinestGrid>>;

  template <typename ElementArray, typename Metavariables,
            typename... InitializationTags>
  static void apply(Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
                    const tuples::TaggedTuple<InitializationTags...>&
                        original_initialization_items) {
    // Copy the initialization items so we can adjust them on each refinement
    // level
    auto initialization_items =
        deserialize<tuples::TaggedTuple<InitializationTags...>>(
            serialize<tuples::TaggedTuple<InitializationTags...>>(
                original_initialization_items)
                .data());
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& element_array =
        Parallel::get_parallel_component<ElementArray>(local_cache);
    const auto& domain = get<domain::Tags::Domain<Dim>>(local_cache);
    auto& initial_refinement_levels =
        get<domain::Tags::InitialRefinementLevels<Dim>>(initialization_items);
    auto& children_refinement_levels =
        get<Tags::ChildrenRefinementLevels<Dim>>(initialization_items);
    auto& parent_refinement_levels =
        get<Tags::ParentRefinementLevels<Dim>>(initialization_items);
    std::optional<size_t> max_levels =
        get<Tags::MaxLevels<OptionsGroup>>(local_cache);
    if (max_levels == 0) {
      ERROR_NO_TRACE(
          "The 'MaxLevels' option includes the finest grid, so '0' is not a "
          "valid value. Set the option to '1' to effectively disable "
          "multigrid.");
    }
    const size_t num_iterations =
        get<Convergence::Tags::Iterations<OptionsGroup>>(local_cache);
    if (UNLIKELY(num_iterations == 0)) {
      Parallel::printf(
          "%s is disabled (zero iterations). Only creating a single grid.\n",
          Options::name<OptionsGroup>());
      max_levels = 1;
    }
    size_t multigrid_level = 0;
    do {
      // Store the current grid as child grid before coarsening it
      children_refinement_levels = initial_refinement_levels;
      initial_refinement_levels = parent_refinement_levels;
      // Construct coarsened (parent) grid
      if (not max_levels.has_value() or multigrid_level < *max_levels - 1) {
        parent_refinement_levels =
            LinearSolver::multigrid::coarsen(initial_refinement_levels);
      }
      // Create element IDs for all elements on this level
      std::vector<ElementId<Dim>> element_ids{};
      for (const auto& block : domain.blocks()) {
        const std::vector<ElementId<Dim>> block_element_ids =
            initial_element_ids(block.id(),
                                initial_refinement_levels[block.id()],
                                multigrid_level);
        element_ids.insert(element_ids.begin(), block_element_ids.begin(),
                           block_element_ids.end());
      }
      // Create an array section for this refinement level
      std::vector<CkArrayIndex> array_indices(element_ids.size());
      std::transform(
          element_ids.begin(), element_ids.end(), array_indices.begin(),
          [](const ElementId<Dim>& local_element_id) {
            return Parallel::ArrayIndex<ElementId<Dim>>(local_element_id);
          });
      using MultigridLevelSection =
          Parallel::Section<ElementArray, Tags::MultigridLevel>;
      get<Parallel::Tags::Section<ElementArray, Tags::MultigridLevel>>(
          initialization_items) = MultigridLevelSection{
          multigrid_level, MultigridLevelSection::cproxy_section::ckNew(
                               element_array.ckGetArrayID(),
                               array_indices.data(), array_indices.size())};
      using FinestGridSection =
          Parallel::Section<ElementArray, Tags::IsFinestGrid>;
      get<Parallel::Tags::Section<ElementArray, Tags::IsFinestGrid>>(
          initialization_items) =
          multigrid_level == 0
              ? std::make_optional(FinestGridSection{
                    true, FinestGridSection::cproxy_section::ckNew(
                              element_array.ckGetArrayID(),
                              array_indices.data(), array_indices.size())})
              : std::nullopt;
      // Create the elements for this refinement level and distribute them among
      // processors
      const int number_of_procs = sys::number_of_procs();
      const domain::BlockZCurveProcDistribution<Dim> element_distribution{
          static_cast<size_t>(number_of_procs), initial_refinement_levels};
      for (const auto& element_id : element_ids) {
        const size_t target_proc =
            element_distribution.get_proc_for_element(element_id);
        element_array(element_id)
            .insert(global_cache, initialization_items, target_proc);
      }
      Parallel::printf(
          "%s level %zu has %zu elements in %zu blocks distributed on %d "
          "procs.\n",
          Options::name<OptionsGroup>(), multigrid_level, element_ids.size(),
          domain.blocks().size(), number_of_procs);
      ++multigrid_level;
    } while (initial_refinement_levels != parent_refinement_levels);
    element_array.doneInserting();
  }
};

}  // namespace LinearSolver::multigrid
