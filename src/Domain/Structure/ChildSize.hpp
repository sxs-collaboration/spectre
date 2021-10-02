// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "Domain/Structure/SegmentId.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"

namespace domain {
/// @{
/*!
 * \brief Size of a child segment relative to its parent
 *
 * Determines which part of the `parent_segment_id` is covered by the
 * `child_segment_id`: The full segment, its lower half or its upper half.
 */
Spectral::ChildSize child_size(const SegmentId& child_segment_id,
                               const SegmentId& parent_segment_id);

template <size_t Dim>
std::array<Spectral::ChildSize, Dim> child_size(
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids);
/// @}
}  // namespace domain
