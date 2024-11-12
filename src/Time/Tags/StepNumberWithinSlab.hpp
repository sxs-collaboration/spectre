// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>

#include "DataStructures/DataBox/Tag.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Number of time step taken within a Slab
struct StepNumberWithinSlab : db::SimpleTag {
  using type = uint64_t;
};
}  // namespace Tags
