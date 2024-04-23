// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"

namespace Parallel::Tags {
/// \brief A counter of the number of elements on the node that have marked
/// themselves as having terminated their current phase.
struct NumberOfElementsTerminated : db::SimpleTag {
  using type = size_t;
};
}  // namespace Parallel::Tags
