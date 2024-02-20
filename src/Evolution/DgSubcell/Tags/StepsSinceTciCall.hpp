// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"

namespace evolution::dg::subcell::Tags {
/// \brief Keeps track of the number of steps since the TCI was called on the
/// FD grid. This is not used on the DG grid.
struct StepsSinceTciCall : db::SimpleTag {
  using type = size_t;
};
}  // namespace evolution::dg::subcell::Tags
