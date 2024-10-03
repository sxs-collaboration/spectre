// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"

namespace Tags {
/// \ingroup TimeGroup
/// \brief Tag forcing a constant step size over a region in an LTS evolution.
struct FixedLtsRatio : db::SimpleTag {
  using type = std::optional<size_t>;
};
}  // namespace Tags
