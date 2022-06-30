// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Amr/Flag.hpp"

namespace amr::domain::Tags {
/// amr::domain::Flag%s for an Element.
template <size_t VolumeDim>
struct Flags : db::SimpleTag {
  using type = std::array<amr::domain::Flag, VolumeDim>;
};

}  // namespace amr::domain::Tags
