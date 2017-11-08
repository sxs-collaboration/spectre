// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementId.hpp"

#include <boost/functional/hash.hpp>
#include <limits>

#include "Domain/ElementIndex.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const size_t block_id) noexcept
    : block_id_{block_id},
      segment_ids_(make_array<VolumeDim>(SegmentId(0, 0))) {}

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(
    const size_t block_id,
    std::array<SegmentId, VolumeDim> segment_ids) noexcept
    : block_id_{block_id}, segment_ids_(std::move(segment_ids)) {}

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const ElementIndex<VolumeDim>& index) noexcept {
  block_id_ = index.block_id();
  for (size_t d = 0; d < VolumeDim; ++d) {
    gsl::at(segment_ids_, d) =
        SegmentId{gsl::at(index.segments(), d).refinement_level(),
                  gsl::at(index.segments(), d).index()};
  }
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::id_of_child(const size_t dim,
                                                       const Side side) const
    noexcept {
  std::array<SegmentId, VolumeDim> new_segment_ids = segment_ids_;
  gsl::at(new_segment_ids, dim) =
      gsl::at(new_segment_ids, dim).id_of_child(side);
  return {block_id_, new_segment_ids};
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::id_of_parent(const size_t dim) const
    noexcept {
  std::array<SegmentId, VolumeDim> new_segment_ids = segment_ids_;
  gsl::at(new_segment_ids, dim) = gsl::at(new_segment_ids, dim).id_of_parent();
  return {block_id_, new_segment_ids};
}

template <size_t VolumeDim>
void ElementId<VolumeDim>::pup(PUP::er& p) noexcept {
  p | block_id_;
  p | segment_ids_;
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const ElementId<VolumeDim>& id) {
  os << "[B" << id.block_id() << ',' << id.segment_ids() << ']';
  return os;
}

// LCOV_EXCL_START
template <size_t VolumeDim>
size_t hash_value(const ElementId<VolumeDim>& c) noexcept {
  size_t h = 0;
  boost::hash_combine(h, c.block_id());
  boost::hash_combine(h, c.segment_ids());
  return h;
}

namespace std {
template <size_t VolumeDim>
size_t hash<ElementId<VolumeDim>>::operator()(
    const ElementId<VolumeDim>& c) const noexcept {
  return hash_value(c);
}
}  // namespace std
// LCOV_EXCL_STOP

template class ElementId<1>;
template class ElementId<2>;
template class ElementId<3>;

template std::ostream& operator<<(std::ostream&, const ElementId<1>&);
template std::ostream& operator<<(std::ostream&, const ElementId<2>&);
template std::ostream& operator<<(std::ostream&, const ElementId<3>&);

template size_t hash_value(const ElementId<1>&) noexcept;
template size_t hash_value(const ElementId<2>&) noexcept;
template size_t hash_value(const ElementId<3>&) noexcept;

namespace std {
template struct hash<ElementId<1>>;
template struct hash<ElementId<2>>;
template struct hash<ElementId<3>>;
}  // namespace std
