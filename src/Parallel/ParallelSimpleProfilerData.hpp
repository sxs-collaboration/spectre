// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Parallel {
class NodeLock;
}  // namespace Parallel
namespace Parallel::Tags {
struct SimpleProfilerData;
}  // namespace Parallel::Tags
namespace profiling {
template <typename Metavariables>
struct SimpleProfilerDataComponent;
class SimpleProfilerData;
}  // namespace profiling
/// \endcond

namespace profiling {
namespace detail {
struct GetParallelSimpleProfilerData {
  using return_type = gsl::not_null<SimpleProfilerData*>;
  template <typename ParallelComponent, typename DbTags>
  static gsl::not_null<SimpleProfilerData*> apply(
      db::DataBox<DbTags>& box,
      const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
      const size_t proc_on_node) {
    auto& all_data =
        db::get_mutable_reference<Parallel::Tags::SimpleProfilerData>(
            make_not_null(&box));
    return &all_data[proc_on_node];
  }
};
}  // namespace detail

/// Get the `SimpleProfilerData` for profiling the current thread.
/// Returns null if there is no `SimpleProfilerDataComponent`.
template <typename Metavariables>
SimpleProfilerData* parallel_simple_profiler_data(
    Parallel::GlobalCache<Metavariables>& cache) {
  if constexpr (Parallel::component_exists_v<
                    Metavariables,
                    SimpleProfilerDataComponent<Metavariables>>) {
    return Parallel::local_synchronous_action<
        detail::GetParallelSimpleProfilerData>(
        Parallel::get_parallel_component<
            SimpleProfilerDataComponent<Metavariables>>(cache),
        static_cast<size_t>(cache.my_local_rank()));
  } else {
    return nullptr;
  }
}
}  // namespace profiling
