// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Domain/Structure/SegmentId.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"

namespace domain {
/*!
 * \brief Size of a child segment relative to its parent
 *
 * Determines which part of the `parent_segment_id` is covered by the
 * `child_segment_id`: The full segment, its lower half or its upper half.
 */
Spectral::ChildSize child_size(const SegmentId& child_segment_id,
                               const SegmentId& parent_segment_id) noexcept;
}
