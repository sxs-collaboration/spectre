// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain {
/// Returns a `std::string` with diagnostic information about how elements and
/// grid points are distributed on the nodes and cores.
template <typename Metavariables>
std::string diagnostic_info(const size_t total_number_of_blocks,
                            const Parallel::GlobalCache<Metavariables>& cache,
                            const std::vector<size_t>& elements_per_core,
                            const std::vector<size_t>& elements_per_node,
                            const std::vector<size_t>& grid_points_per_core,
                            const std::vector<size_t>& grid_points_per_node) {
  const size_t total_number_of_elements =
      alg::accumulate(elements_per_node, 0_st);
  const size_t total_number_of_grid_points =
      alg::accumulate(grid_points_per_node, 0_st);
  const size_t number_of_cores = Parallel::number_of_procs<size_t>(cache);
  const size_t number_of_nodes = Parallel::number_of_nodes<size_t>(cache);

  // Sanity checks
  ASSERT(total_number_of_elements == alg::accumulate(elements_per_core, 0_st),
         "Number of elements determined from elements_per_core ("
             << alg::accumulate(elements_per_core, 0_st)
             << ") does not match number of elements determined from "
                "elements_per_node ("
             << total_number_of_elements << ").");
  ASSERT(total_number_of_grid_points ==
             alg::accumulate(grid_points_per_core, 0_st),
         "Number of grid points determined from grid_points_per_core ("
             << alg::accumulate(grid_points_per_core, 0_st)
             << ") does not match number of grid points determined from "
                "grid points_per_node ("
             << total_number_of_grid_points << ").");
  ASSERT(number_of_cores == elements_per_core.size(),
         "Number of cores determined from elements_per_core ("
             << elements_per_core.size()
             << ") does not match number of cores determined from "
                "global cache ("
             << number_of_cores << ").");
  ASSERT(number_of_cores == grid_points_per_core.size(),
         "Number of cores determined from grid_points_per_core ("
             << grid_points_per_core.size()
             << ") does not match number of cores determined from "
                "global cache ("
             << number_of_cores << ").");
  ASSERT(number_of_nodes == elements_per_node.size(),
         "Number of nodes determined from elements_per_node ("
             << elements_per_node.size()
             << ") does not match number of nodes determined from "
                "global cache ("
             << number_of_nodes << ").");
  ASSERT(number_of_nodes == grid_points_per_node.size(),
         "Number of nodes determined from grid_points_per_node ("
             << grid_points_per_node.size()
             << ") does not match number of nodes determined from "
                "global cache ("
             << number_of_nodes << ").");

  using ::operator<<;
  std::stringstream ss{};
  ss << "----- Domain Info -----\n"
     << "Total blocks: " << total_number_of_blocks << "\n"
     << "Total elements: " << total_number_of_elements << "\n"
     << "Total grid points: " << total_number_of_grid_points << "\n"
     << "Number of cores: " << number_of_cores << "\n"
     << "Number of nodes: " << number_of_nodes << "\n"
     << "Elements per core: " << elements_per_core << "\n"
     << "Elements per node: " << elements_per_node << "\n"
     << "Grid points per core: " << grid_points_per_core << "\n"
     << "Grid points per node: " << grid_points_per_node << "\n"
     << "-----------------------\n";

  return ss.str();
}
}  // namespace domain
