// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <unordered_set>

#include "Utilities/TaggedTuple.hpp"

#include "Parallel/GlobalCache.decl.h"

namespace TestHelpers::Parallel {
/// \brief Used to assign elements to an array component.
///
/// \details `ArrayIndex` can be used if the array index is something other than
/// an int and can be indexed by the `i`th element (where `i` is a size_t).
template <typename Proxy, typename Metavariables, typename... InitTags,
          typename ArrayIndex = int>
void assign_array_elements_round_robin_style(
    Proxy& array_proxy, const size_t num_elements, const size_t num_procs,
    const tuples::TaggedTuple<InitTags...>& initialization_items,
    ::Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
    const std::unordered_set<size_t>& procs_to_ignore,
    const ArrayIndex /*meta*/ = {}) {
  size_t which_proc = 0;
  for (size_t i = 0; i < num_elements; i++) {
    ArrayIndex index{};
    if constexpr (std::is_same_v<ArrayIndex, int>) {
      index = static_cast<int>(i);
    } else {
      index = ArrayIndex{i};
    }
    while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
      which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
    }
    array_proxy[index].insert(global_cache, initialization_items, which_proc);
    which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
  }

  array_proxy.doneInserting();
}
}  // namespace TestHelpers::Parallel
