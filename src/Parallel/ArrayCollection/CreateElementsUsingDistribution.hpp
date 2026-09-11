// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Parallel/DomainDiagnosticInfo.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Numeric.hpp"

/// \cond
namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace Parallel {
/*!
 * \brief Creates elements using a chosen distribution.
 *
 * The `func` is called with `(element_id, target_proc, target_node)` allowing
 * the `func` to insert the element with `element_id` on the target processor
 * and node.
 *
 * The `weighting_extents_override`, if present, gives per-block extents to use
 * instead of `initial_extents` only when computing the cost of elements for
 * the distribution (e.g. to weight subcell-capable elements by their
 * finite-difference grid rather than their DG grid). Only blocks whose
 * weighting extents differ from `initial_extents` need an entry; blocks not
 * found in the map use `initial_extents`. The actual `initial_extents` are
 * always used for everything else, e.g. the diagnostics below.
 */
template <typename F, size_t Dim, typename Metavariables>
void create_elements_using_distribution(
    const F& func, const std::optional<domain::ElementWeight>& element_weight,
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const Spectral::Basis i1_basis, const Spectral::Quadrature i1_quadrature,

    const std::unordered_set<size_t>& procs_to_ignore,
    const size_t number_of_procs, const size_t number_of_nodes,
    const size_t num_of_procs_to_use,
    const Parallel::GlobalCache<Metavariables>& local_cache,
    const bool print_diagnostics,
    const std::optional<std::unordered_map<size_t, std::array<size_t, Dim>>>&
        weighting_extents_override = std::nullopt) {
  // Only need the element distribution if the element weight has a value
  // because then we have to use the space filling curve and not just use round
  // robin.
  domain::BlockZCurveProcDistribution<Dim> element_distribution{};
  if (element_weight.has_value()) {
    std::optional<std::vector<std::array<size_t, Dim>>>
        weighting_extents_buffer{};
    if (weighting_extents_override.has_value()) {
      weighting_extents_buffer = initial_extents;
      const size_t num_blocks = weighting_extents_buffer->size();
      for (const auto& block_id_and_extents :
           weighting_extents_override.value()) {
        ASSERT(block_id_and_extents.first < num_blocks,
               "The weighting extents override contains block ID "
                   << block_id_and_extents.first << " but there are only "
                   << num_blocks << " blocks.");
        (*weighting_extents_buffer)[block_id_and_extents.first] =
            block_id_and_extents.second;
      }
    }
    const std::vector<std::array<size_t, Dim>>& weighting_extents =
        weighting_extents_buffer.has_value() ? weighting_extents_buffer.value()
                                             : initial_extents;
    const std::unordered_map<ElementId<Dim>, double> element_costs =
        domain::get_element_costs(blocks, initial_refinement_levels,
                                  weighting_extents, element_weight.value(),
                                  i1_basis, i1_quadrature);
    // Only the element costs depend on the weighting extents. The distribution
    // itself just ASSERTs that `initial_extents` has one entry per block, so
    // pass the true extents here.
    element_distribution = domain::BlockZCurveProcDistribution<Dim>{
        element_costs,   num_of_procs_to_use, blocks, initial_refinement_levels,
        initial_extents, procs_to_ignore};
  }

  // Will be used to print domain diagnostic info
  std::vector<size_t> elements_per_core(number_of_procs, 0_st);
  std::vector<size_t> elements_per_node(number_of_nodes, 0_st);
  std::vector<size_t> grid_points_per_core(number_of_procs, 0_st);
  std::vector<size_t> grid_points_per_node(number_of_nodes, 0_st);

  size_t which_proc = 0;
  for (const auto& block : blocks) {
    const auto& initial_ref_levs = initial_refinement_levels[block.id()];
    const size_t grid_points_per_element =
        alg::accumulate(initial_extents[block.id()], 1_st, std::multiplies<>());

    const std::vector<ElementId<Dim>> element_ids =
        initial_element_ids(block.id(), initial_ref_levs);

    // Value means ZCurve. nullopt means round robin
    if (element_weight.has_value()) {
      for (const auto& element_id : element_ids) {
        const size_t target_proc =
            element_distribution.get_proc_for_element(element_id);
        const size_t target_node =
            Parallel::node_of<size_t>(target_proc, local_cache);
        func(element_id, target_proc, target_node);

        ++elements_per_core[target_proc];
        ++elements_per_node[target_node];
        grid_points_per_core[target_proc] += grid_points_per_element;
        grid_points_per_node[target_node] += grid_points_per_element;
      }
    } else {
      for (size_t i = 0; i < element_ids.size(); ++i) {
        while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
          which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
        }
        const size_t target_proc = which_proc;
        const size_t target_node =
            Parallel::node_of<size_t>(which_proc, local_cache);
        const ElementId<Dim> element_id(element_ids[i]);
        func(element_id, target_proc, target_node);

        ++elements_per_core[which_proc];
        ++elements_per_node[target_node];
        grid_points_per_core[which_proc] += grid_points_per_element;
        grid_points_per_node[target_node] += grid_points_per_element;

        which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
      }
    }
  }

  if (print_diagnostics) {
    Parallel::printf("\n%s\n", domain::diagnostic_info(
                                   blocks.size(), local_cache,
                                   elements_per_core, elements_per_node,
                                   grid_points_per_core, grid_points_per_node));
  }
}
}  // namespace Parallel
