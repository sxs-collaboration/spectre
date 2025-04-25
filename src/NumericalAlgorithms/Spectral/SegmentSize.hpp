// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <iosfwd>

namespace Spectral {

/// \brief The portion of a Segment (or the interval \f$[-1, 1]\f$)
/// covered by a mesh.
///
/// \details SegmentSize is used to denote the relationship between two meshes
/// (in each dimension) such as:
/// - a face mesh and a mortar mesh when computing boundary corrections on the
///   mortars between neighboring elements.  In this case the face mesh will
///   always have SegmentSize::Full, and the mortar mesh can have any value.
/// - a parent mesh and a child mesh in a h-refinement hierarchy.  In this case
///   the parent mesh will have SegmentSize::Full, while the child mesh will
///   have either SegmentSize::UpperHalf or SegmentSize::LowerHalf.
/// - for p-preojection, the source and target meshes should use
///   SegmentSize::Full
enum class SegmentSize : uint8_t {
  Uninitialized = 0,
  Full = 1,
  UpperHalf = 2,
  LowerHalf = 3
};

std::ostream& operator<<(std::ostream& os, SegmentSize segment_size);
}  // namespace Spectral
