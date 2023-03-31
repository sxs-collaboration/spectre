// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Domain/Structure/Side.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Neighbors;
class SegmentId;
namespace gsl {
template <typename T>
class not_null;
}
/// \endcond

namespace TestHelpers::domain {

/// The type of face of an Element
enum class FaceType {
  /// Used to denote an Element face on an external boundary
  External,
  /// Used to denote an Element face on a periodic boundary
  Periodic,
  /// Used to denote an Element face shared with a neighboring Block
  Block,
  /// Used to denote an Element face not on the boundary of a Block
  Internal
};

/// Valid SegmentIds for a neighbor of the root segment on the given side
///
/// \details Returns an empty vector if `side` abuts an external boundary
/// (indicated by passing External as the `face_type`). Returns the root segment
/// if `face_type` is Periodic.
///
/// \note If the neighbor segment is in another Block (indicated by passing
/// Block as `face_type`), the returned SegmentId%s are in the logical frame of
/// `segment_id` (i.e. they will need to be mapped to the neighbor logical frame
/// by using the appropriate OrientationMap of the Block owning the ElementId
/// with `segment_id`).
std::vector<SegmentId> valid_neighbor_segments(const FaceType face_type,
                                               const Side side);

/// Valid SegmentIds for a neighbor on the sibling side of a segment
std::vector<SegmentId> valid_neighbor_segments(const SegmentId& segment_id);

/// Valid SegmentIds for a neighbor on the non-sibling side of a segment
///
/// \details Returns an empty vector if the non-sibling side abuts an external
/// boundary (indicated by passing External as `face_type`).
///
/// \note If the non-sibling side abuts another Block (indicated by passing
/// Block as `face_type`), the returned SegmentId%s are in the logical frame of
/// `segment_id` (i.e. they will need to be mapped to the neighbor logical frame
/// by using the appropriate OrientationMap of the Block owning the ElementId
/// with `segment_id`).
std::vector<SegmentId> valid_neighbor_segments(const SegmentId& segment_id,
                                               const FaceType face_type);

/// Valid Neighbors for an Element with `element_id` in the given `direction`
/// with the given `face_type`
///
/// \details FaceType needs to be passed in only if the Element abuts a Block
/// boundary in the given `direction`
template <size_t Dim>
std::vector<Neighbors<Dim>> valid_neighbors(
    gsl::not_null<std::mt19937*> generator, const ElementId<Dim>& element_id,
    const Direction<Dim>& direction,
    const FaceType face_type = FaceType::Internal);

/// Checks that the `neighbors` for the Element with `element_id` in the given
/// `direction` are a complete set of valid face neighbors
///
/// \details A valid neighbor is within one refinement level in the dimensions
/// parallel to the face between the Element  and the neighbor.  The set is
/// complete if there are neighboring Element%s they completely cover the face.
template <size_t Dim>
void check_neighbors(const Neighbors<Dim>& neighbors,
                     const ElementId<Dim>& element_id,
                     const Direction<Dim>& direction);
}  // namespace TestHelpers::domain
