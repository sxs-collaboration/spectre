// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Amr/Flag.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace amr::Tags {
/// amr::Flag%s for the neighbors of an Element.
template <size_t VolumeDim>
struct NeighborFlags : db::SimpleTag {
  using type = std::unordered_map<ElementId<VolumeDim>,
                                  std::array<amr::Flag, VolumeDim>>;
};

}  // namespace amr::Tags
