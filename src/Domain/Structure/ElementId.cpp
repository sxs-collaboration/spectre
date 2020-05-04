// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ElementId.hpp"

#include <boost/functional/hash.hpp>
#include <limits>
#include <ostream>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

// The `static_assert`s verify that `ElementId` satisfies the constraints
// imposed by Charm++ to make `ElementId` able to act as an index into Charm++'s
// arrays. These constraints are:
// - `ElementId` must satisfy `std::is_pod`
// - `ElementId` must not be larger than the size of three `int`s, i.e.
//   `sizeof(ElementId) <= 3 * sizeof(int)`
static_assert(std::is_pod<ElementId<1>>::value, "ElementId is not POD");
static_assert(std::is_pod<ElementId<2>>::value, "ElementId is not POD");
static_assert(std::is_pod<ElementId<3>>::value, "ElementId is not POD");

static_assert(sizeof(ElementId<1>) == 1 * sizeof(int),
              "Wrong size for ElementId<1>");
static_assert(sizeof(ElementId<2>) == 2 * sizeof(int),
              "Wrong size for ElementId<2>");
static_assert(sizeof(ElementId<3>) == 3 * sizeof(int),
              "Wrong size for ElementId<3>");

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const size_t block_id) noexcept
    : segment_ids_(make_array<VolumeDim>(SegmentId(block_id, 0, 0))) {}

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(
    const size_t block_id,
    std::array<SegmentId, VolumeDim> segment_ids) noexcept
    : segment_ids_(segment_ids) {
  for (size_t d = 0; d < VolumeDim; ++d) {
    gsl::at(segment_ids_, d).set_block_id(block_id);
  }
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::id_of_child(const size_t dim,
                                                       const Side side) const
    noexcept {
  std::array<SegmentId, VolumeDim> new_segment_ids = segment_ids_;
  gsl::at(new_segment_ids, dim) =
      gsl::at(new_segment_ids, dim).id_of_child(side);
  return {block_id(), new_segment_ids};
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::id_of_parent(const size_t dim) const
    noexcept {
  std::array<SegmentId, VolumeDim> new_segment_ids = segment_ids_;
  gsl::at(new_segment_ids, dim) = gsl::at(new_segment_ids, dim).id_of_parent();
  return {block_id(), new_segment_ids};
}

template <size_t VolumeDim>
void ElementId<VolumeDim>::pup(PUP::er& p) noexcept {
  p | segment_ids_;
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::external_boundary_id() noexcept {
  // We use the maximum possible value that can be stored in `block_id_bits` and
  // the maximum refinement level to signal an external boundary. While in
  // theory this could cause a problem if we have an element at the highest
  // refinement level in the largest block (by id), in practice it is very
  // unlikely we will ever get to that large of a refinement.
  return ElementId<VolumeDim>(
      two_to_the(SegmentId::block_id_bits) - 1,
      make_array<VolumeDim>(SegmentId(SegmentId::max_refinement_level, 0)));
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const ElementId<VolumeDim>& id) noexcept {
  os << "[B" << id.block_id() << ',' << id.segment_ids() << ']';
  return os;
}

// LCOV_EXCL_START
template <size_t VolumeDim>
size_t hash_value(const ElementId<VolumeDim>& id) noexcept {
  // ElementId is used as an opaque array of bytes by Charm, so we
  // treat it that way as well.
  // clang-tidy: do not use reinterpret_cast
  const auto bytes = reinterpret_cast<const char*>(&id);  // NOLINT
  // clang-tidy: do not use pointer arithmetic
  return boost::hash_range(bytes, bytes + sizeof(id));  // NOLINT
}

// clang-tidy: do not modify namespace std
namespace std {  // NOLINT
template <size_t VolumeDim>
size_t hash<ElementId<VolumeDim>>::operator()(
    const ElementId<VolumeDim>& id) const noexcept {
  return hash_value(id);
}
}  // namespace std
// LCOV_EXCL_STOP

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template class ElementId<GET_DIM(data)>;                                     \
  template std::ostream& operator<<(std::ostream&,                             \
                                    const ElementId<GET_DIM(data)>&) noexcept; \
  template size_t hash_value(const ElementId<GET_DIM(data)>&) noexcept;        \
  namespace std { /* NOLINT */                                                 \
  template struct hash<ElementId<GET_DIM(data)>>;                              \
  }

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
