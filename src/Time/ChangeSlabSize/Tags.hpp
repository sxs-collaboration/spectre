// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/TimeStepRequestProcessor.hpp"

namespace Tags::ChangeSlabSize {
/// Sizes requested for each slab by ChangeSlabSize events.
struct NewSlabSize : db::SimpleTag {
  using type = std::map<int64_t, std::vector<TimeStepRequestProcessor>>;
};

/// Number of ChangeSlabSize events changing the size at each slab.
struct NumberOfExpectedMessages : db::SimpleTag {
  using type = std::map<int64_t, size_t>;
};

/// Long-term desired slab size.  Used as the default size if nothing
/// chooses a smaller one.
struct SlabSizeGoal : db::SimpleTag {
  using type = double;
};
}  // namespace Tags::ChangeSlabSize
