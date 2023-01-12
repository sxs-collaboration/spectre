// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Structure/ElementId.hpp"

namespace domain {
/*!
 * \brief Constructs the affine map from ElementLogical to BlockLogical
 * coordinates
 *
 * An element is the result of repeatedly splitting a block in half along any of
 * its logical axes. This map transforms from ElementLogical coordinates
 * [-1, 1]^dim to the subset of BlockLogical coordinates [-1, 1]^dim that cover
 * the element. For instance, the two elements at refinement level 1 in 1D
 * cover [-1, 0] and [0, 1] in BlockLogical coordinates, respectively.
 */
template <size_t Dim>
std::unique_ptr<
    CoordinateMapBase<Frame::ElementLogical, Frame::BlockLogical, Dim>>
element_to_block_logical_map(const ElementId<Dim>& element_id);
}  // namespace domain
