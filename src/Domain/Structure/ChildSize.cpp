// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ChildSize.hpp"

#include <ostream>
#include <string>

#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace domain {

Spectral::ChildSize child_size(const SegmentId& child_segment_id,
                               const SegmentId& parent_segment_id) noexcept {
  if (child_segment_id == parent_segment_id) {
    return Spectral::ChildSize::Full;
  } else {
    ASSERT(child_segment_id.id_of_parent() == parent_segment_id,
           "Segment id '" << parent_segment_id << "' is not the parent of '"
                          << child_segment_id << "'.");
    return parent_segment_id.id_of_child(Side::Lower) == child_segment_id
               ? Spectral::ChildSize::LowerHalf
               : Spectral::ChildSize::UpperHalf;
  }
}

}  // namespace domain
