// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"

namespace evolution::dg::subcell::Tags {
/// \brief Keeps track of the number of times the TCI was called after a
/// rollback.
struct TciCallsSinceRollback : db::SimpleTag {
  using type = size_t;
};
}  // namespace evolution::dg::subcell::Tags
