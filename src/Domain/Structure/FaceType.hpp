// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <iosfwd>

namespace domain {

/// \brief  The type of face for a Block or Element
///
/// \details For a face with neighboring Blocks (or Elements) the FaceType is
/// determined by whether the block logical coordinates of the  neighboring
/// Blocks (or Elements) have conforming block logical coordinates on their
/// interface. Block logical coordinates are considered to be conforming if
/// they are identical (i.e. an aligned OrientationMap) or related by a
/// discrete rotation (i.e. a valid non-aligned OrientationMap).
///
/// \note FaceType::Topological is for the faces of an Element or Block with a
/// Topology other than I1 (e.g. the angular directions of an S2 topology). The
/// face is not on the external boundary, and has no neighboring Block or
/// Element.
enum class FaceType : uint8_t {
  /// Used to denote an uninitialized value
  Uninitialized = 0,
  /// Used to denote a face on an external boundary
  External,
  /// Used to denote a face that is not on the external boundary, and that has
  /// no neighboring Block or Element (e.g. the angular directions of an S2
  /// topology).
  Topological,
  /// Used to denote a face shared with one or more conforming, aligned
  /// neighbors
  ConformingAligned,
  /// Used to denote a face shared with one or more conforming, unaligned
  /// neighbors
  ConformingUnaligned,
  /// Used to denote a face shared with a single non-conforming neighbor
  SingleNonconforming,
  /// Used to denote a face shared with multiple non-conforming neighbors
  MultipleNonconforming,
};

/// Output operator for a FaceType.
std::ostream& operator<<(std::ostream& os, FaceType face_type);
}  // namespace domain
