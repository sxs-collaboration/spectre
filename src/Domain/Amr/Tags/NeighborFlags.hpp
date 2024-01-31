// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"

/// \cond
namespace amr {
template <size_t VolumeDim>
struct Info;
}  // namespace amr
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace amr::Tags {
/// \brief amr::Info for the neighbors of an Element.
///
/// Note that the `amr::Info` is stored in the orientation of the neighbor,
/// i.e., dimensions are not reoriented to the orientation of the element.
template <size_t VolumeDim>
struct NeighborInfo : db::SimpleTag {
  using type = std::unordered_map<ElementId<VolumeDim>, amr::Info<VolumeDim>>;
};

}  // namespace amr::Tags
