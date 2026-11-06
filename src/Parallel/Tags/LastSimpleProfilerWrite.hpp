// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>

#include "DataStructures/DataBox/Tag.hpp"

namespace Parallel::Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the last profiling observation value from the DataBox.
struct LastSimpleProfilerWrite : db::SimpleTag {
  using type = std::atomic<double>;
};
}  // namespace Parallel::Tags
