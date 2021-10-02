// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ChildSize.hpp"

#include <array>
#include <cstddef>
#include <ostream>
#include <string>

#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {

Spectral::ChildSize child_size(const SegmentId& child_segment_id,
                               const SegmentId& parent_segment_id) {
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

template <size_t Dim>
std::array<Spectral::ChildSize, Dim> child_size(
    const std::array<SegmentId, Dim>& child_segment_ids,
    const std::array<SegmentId, Dim>& parent_segment_ids) {
  std::array<Spectral::ChildSize, Dim> result{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = child_size(gsl::at(child_segment_ids, d),
                                    gsl::at(parent_segment_ids, d));
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template std::array<Spectral::ChildSize, DIM(data)> child_size( \
      const std::array<SegmentId, DIM(data)>& child_segment_ids,  \
      const std::array<SegmentId, DIM(data)>& parent_segment_ids);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace domain
