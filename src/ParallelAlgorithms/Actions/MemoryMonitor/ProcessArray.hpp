// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Numeric.hpp"

namespace mem_monitor {
/*!
 * \brief Simple action meant to be used as a callback for
 * Parallel::contribute_to_reduction that writes the size of an Array parallel
 * component to disk.
 *
 * \details The columns in the dat file when running on 3 nodes will be
 *
 * - %Time
 * - Size on node 0 (MB)
 * - Size on node 1 (MB)
 * - Size on node 2 (MB)
 * - Average size per node (MB)
 *
 * The dat file will be placed in the `/MemoryMonitors/` group in the reduction
 * file. The name of the dat file is the `pretty_type::name` of the component.
 */
template <typename ArrayComponent>
struct ProcessArray {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const double time,
                    const std::vector<double>& size_per_node) {
    auto& observer_writer_proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    std::vector<std::string> legend{{"Time"}};
    for (size_t i = 0; i < size_per_node.size(); i++) {
      legend.emplace_back("Size on node " + get_output(i) + " (MB)");
    }
    legend.emplace_back("Average size per node (MB)");

    const double avg_size = alg::accumulate(size_per_node, 0.0) /
                            static_cast<double>(size_per_node.size());

    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        // Node 0 is always the writer
        observer_writer_proxy[0], subfile_name<ArrayComponent>(), legend,
        std::make_tuple(time, size_per_node, avg_size));
  }
};
}  // namespace mem_monitor
