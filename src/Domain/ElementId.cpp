// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementId.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>

#include "Domain/ElementIndex.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

namespace domain {
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

// clang-tidy: mark explicit, we want implicit conversion
template <size_t VolumeDim>
ElementId<VolumeDim>::
operator Parallel::ArrayIndex<ElementIndex<VolumeDim>>()  // NOLINT
    const {
  return {ElementIndex<VolumeDim>{*this}};
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
  ::operator<<(os << "[B" << id.block_id() << ',', id.segment_ids()) << ']';
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
}  // namespace domain

// clang-tidy: do not modify namespace std
namespace std {  // NOLINT
template <size_t VolumeDim>
size_t hash<domain::ElementId<VolumeDim>>::operator()(
    const domain::ElementId<VolumeDim>& c) const noexcept {
  return hash_value(c);
}
}  // namespace std
// LCOV_EXCL_STOP

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template class domain::ElementId<DIM(data)>;                      \
  template std::ostream& domain::operator<<(                        \
      std::ostream& os, const domain::ElementId<DIM(data)>& block); \
  template size_t domain::hash_value(                               \
      const domain::ElementId<DIM(data)>&) noexcept;                \
  namespace std { /* NOLINT */                                      \
  template struct hash<domain::ElementId<DIM(data)>>;               \
  }

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
