// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <numeric>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/MemoryMonitor/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace mem_monitor {
/*!
 * \brief Simple action meant to be run on the MemoryMonitor component that
 * collects sizes from Groups and Nodegroups.
 *
 * \details This action collects the sizes of all the local branches of a group
 * or nodegroup component, computes the total memory usage on a node for each,
 * then writes it to disk. For groups, the proc with the maximum memory usage is
 * also reported along with the size on the proc.
 *
 * The columns in the dat file for a nodegroup when running on 3 nodes will be
 *
 * - %Time
 * - Size on node 0 (MB)
 * - Size on node 1 (MB)
 * - Size on node 2 (MB)
 * - Average size per node (MB)
 *
 * The columns in the dat file for a group when running on 3 nodes will be
 *
 * - %Time
 * - Size on node 0 (MB)
 * - Size on node 1 (MB)
 * - Size on node 2 (MB)
 * - Proc of max size
 * - Size on proc of max size (MB)
 * - Average size per node (MB)
 *
 * The dat file will be placed in the `/MemoryMonitors/` group in the reduction
 * file. The name of the dat file is the `pretty_type::name` of the component.
 */
template <typename ContributingComponent>
struct ContributeMemoryData {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const double time,
                    const int node_or_proc, const double size_in_megabytes) {
    static_assert(Parallel::is_group_v<ContributingComponent> or
                  Parallel::is_nodegroup_v<ContributingComponent>);

    using tag = Tags::MemoryHolder;
    db::mutate<tag>(
        [&cache, &time, &node_or_proc, &size_in_megabytes](
            const gsl::not_null<std::unordered_map<
                std::string,
                std::unordered_map<double, std::unordered_map<int, double>>>*>
                memory_holder_all) {
          auto memory_holder_pair = memory_holder_all->try_emplace(
              pretty_type::name<ContributingComponent>());
          auto& memory_holder = (*memory_holder_pair.first).second;

          memory_holder.try_emplace(time);
          memory_holder.at(time)[node_or_proc] = size_in_megabytes;

          // If we have received data for every node/proc at a given
          // time, get all the data, write it to disk, then remove the current
          // time from the stored times as it's no longer needed

          auto& mem_monitor_proxy =
              Parallel::get_parallel_component<MemoryMonitor<Metavariables>>(
                  cache);

          constexpr bool is_group = Parallel::is_group_v<ContributingComponent>;

          const size_t num_nodes = Parallel::number_of_nodes<size_t>(
              *Parallel::local(mem_monitor_proxy));
          const size_t num_procs = Parallel::number_of_procs<size_t>(
              *Parallel::local(mem_monitor_proxy));
          const size_t expected_number = is_group ? num_procs : num_nodes;
          ASSERT(memory_holder.at(time).size() <= expected_number,
                 "ContributeMemoryData received more data than it was "
                 "expecting. Was expecting "
                     << expected_number << " calls but instead got "
                     << memory_holder.at(time).size());
          if (memory_holder.at(time).size() == expected_number) {
            // First column is always time
            std::vector<double> data_to_append{time};
            std::vector<std::string> legend{{"Time"}};

            // Append a column for each node, and keep track of cumulative
            // total. If we have proc data (from groups) do an additional loop
            // over the procs to get the total on that node and get the proc
            // of the maximum memory usage
            double avg_size_per_node = 0.0;
            double max_usage_on_proc = -std::numeric_limits<double>::max();
            int proc_of_max = 0;
            for (size_t node = 0; node < num_nodes; node++) {
              double size_on_node = 0.0;
              if (not is_group) {
                size_on_node = memory_holder.at(time).at(node);
              } else {
                const int first_proc = Parallel::first_proc_on_node<int>(
                    node, *Parallel::local(mem_monitor_proxy));
                const int procs_on_node = Parallel::procs_on_node<int>(
                    node, *Parallel::local(mem_monitor_proxy));
                const int last_proc = first_proc + procs_on_node;
                for (int proc = first_proc; proc < last_proc; proc++) {
                  size_on_node += memory_holder.at(time).at(proc);
                  if (memory_holder.at(time).at(proc) > max_usage_on_proc) {
                    max_usage_on_proc = memory_holder.at(time).at(proc);
                    proc_of_max = proc;
                  }
                }
              }

              data_to_append.push_back(size_on_node);
              avg_size_per_node += size_on_node;
              legend.emplace_back("Size on node " + get_output(node) + " (MB)");
            }

            // If we have proc data, write the proc with the maximum usage to
            // disk along with how much memory it's using
            if (is_group) {
              data_to_append.push_back(static_cast<double>(proc_of_max));
              data_to_append.push_back(max_usage_on_proc);
              legend.emplace_back("Proc of max size");
              legend.emplace_back("Size on proc of max size (MB)");
            }

            avg_size_per_node /= static_cast<double>(num_nodes);

            // Last column is average over all nodes
            data_to_append.push_back(avg_size_per_node);
            legend.emplace_back("Average size per node (MB)");

            auto& observer_writer_proxy = Parallel::get_parallel_component<
                observers::ObserverWriter<Metavariables>>(cache);

            Parallel::threaded_action<
                observers::ThreadedActions::WriteReductionDataRow>(
                // Node 0 is always the writer
                observer_writer_proxy[0], subfile_name<ContributingComponent>(),
                legend, std::make_tuple(data_to_append));

            // Clean up finished time
            auto finished_time_iter = memory_holder.find(time);
            memory_holder.erase(finished_time_iter);
          }
        },
        make_not_null(&box));
  }
};
}  // namespace mem_monitor
