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
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace mem_monitor {
/*!
 * \brief Simple action meant to be run on a Singleton that writes the size of
 * the Singleton component to disk.
 *
 * \details The columns in the dat file are
 *
 * - %Time
 * - Proc
 * - Size (MB)
 *
 * The dat file will be placed in the `/MemoryMonitors/` group in the reduction
 * file. The name of the dat file is the `pretty_type::name` of the component.
 */
struct ProcessSingleton {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const double time) {
    static_assert(
        Parallel::is_singleton_v<ParallelComponent>,
        "ProcessSingleton can only be run on a Singleton parallel component.");
    auto& singleton_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const double size_in_bytes = static_cast<double>(
        size_of_object_in_bytes(*Parallel::local(singleton_proxy)));
    const double size_in_MB = size_in_bytes / 1.0e6;

    const std::vector<std::string> legend{{"Time", "Proc", "Size (MB)"}};

    auto& observer_writer_proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        // Node 0 is always the writer
        observer_writer_proxy[0], subfile_name<ParallelComponent>(), legend,
        std::make_tuple(
            time, Parallel::my_proc<size_t>(*Parallel::local(singleton_proxy)),
            size_in_MB));
  }
};
}  // namespace mem_monitor
