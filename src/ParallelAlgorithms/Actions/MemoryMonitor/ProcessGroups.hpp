// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/MemoryMonitor/MemoryMonitor.hpp"
#include "Parallel/TypeTraits.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace mem_monitor {
/*!
 * \brief Simple action meant to be run on every branch of a Group or NodeGroup
 * that computes the size of the local branch and reports that size to the
 * MemoryMonitor using the ContributeMemoryData simple action.
 */
struct ProcessGroups {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index, const double time) {
    static_assert(Parallel::is_group_v<ParallelComponent> or
                      Parallel::is_nodegroup_v<ParallelComponent>,
                  "ProcessGroups can only be run on Group or Nodegroup "
                  "parallel components.");
    static_assert(std::is_same_v<ArrayIndex, int>,
                  "ArrayIndex of Group or Nodegroup parallel components must "
                  "be an int to use the ProcessGroups action.");

    auto& singleton_proxy =
        Parallel::get_parallel_component<MemoryMonitor<Metavariables>>(cache);

    auto& group_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    const double size_in_bytes = static_cast<double>(
        size_of_object_in_bytes(*Parallel::local_branch(group_proxy)));

    const double size_in_MB = size_in_bytes / 1.0e6;

    // Note that we don't call Parallel::contribute_to_reduction here. This is
    // because Charm requires that all calls to contribute_to_reduction happen
    // in the exact same order every time, otherwise it is undefined behavior.
    // However, this simple action (ProcessGroups) is called on each branch
    // of a group or nodegroup. Because this is a simple action, the order that
    // the branches run the simple actions is completely random based on
    // communication patterns in charm. Thus, calling contribute_to_reduction
    // here would result in undefined behavior.
    // Also note that `array_index` here is my_node for nodegroups and my_proc
    // for groups
    Parallel::simple_action<ContributeMemoryData<ParallelComponent>>(
        singleton_proxy, time, array_index, size_in_MB);
  }
};
}  // namespace mem_monitor
