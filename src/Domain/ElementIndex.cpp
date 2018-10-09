// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementIndex.hpp"

#include <boost/functional/hash.hpp>
#include <cstring>
#include <ostream>
#include <type_traits>

#include "Domain/ElementId.hpp"  // IWYU pragma: keep
#include "Domain/SegmentId.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

SegmentIndex::SegmentIndex(size_t block_id,
                           const SegmentId& segment_id) noexcept
    : block_id_(block_id),
      refinement_level_(segment_id.refinement_level()),
      index_(segment_id.index()) {
  ASSERT(block_id < two_to_the(ElementIndex_detail::block_id_bits),
         "Block id out of bounds: " << block_id);
  ASSERT(segment_id.refinement_level() <=
             ElementIndex_detail::max_refinement_level,
         "Refinement level out of bounds: " << segment_id.refinement_level());
}

static_assert(std::is_pod<SegmentIndex>::value, "SegmentIndex is not POD");
static_assert(sizeof(SegmentIndex) == sizeof(int),
              "SegmentIndex does not fit in an int");

std::ostream& operator<<(std::ostream& s, const SegmentIndex& index) noexcept {
  return s << '[' << index.block_id() << ':' << index.refinement_level() << ':'
           << index.index() << ']';
}

template <size_t VolumeDim>
ElementIndex<VolumeDim>::ElementIndex(const ElementId<VolumeDim>& id) noexcept {
  for (size_t d = 0; d < VolumeDim; ++d) {
    gsl::at(segments_, d) =
        SegmentIndex(id.block_id(), gsl::at(id.segment_ids(), d));
  }
}

static_assert(std::is_pod<ElementIndex<1>>::value, "ElementIndex is not POD");
static_assert(std::is_pod<ElementIndex<2>>::value, "ElementIndex is not POD");
static_assert(std::is_pod<ElementIndex<3>>::value, "ElementIndex is not POD");

static_assert(sizeof(ElementIndex<1>) == 1 * sizeof(int),
              "Wrong size for ElementIndex<1>");
static_assert(sizeof(ElementIndex<2>) == 2 * sizeof(int),
              "Wrong size for ElementIndex<2>");
static_assert(sizeof(ElementIndex<3>) == 3 * sizeof(int),
              "Wrong size for ElementIndex<3>");

template <size_t VolumeDim>
bool operator==(const ElementIndex<VolumeDim>& a,
                const ElementIndex<VolumeDim>& b) noexcept {
  // ElementIndex is used as an opaque array of bytes by Charm, so we
  // treat it that way as well.
  return 0 == std::memcmp(&a, &b, sizeof(a));
}

template <size_t VolumeDim>
bool operator!=(const ElementIndex<VolumeDim>& a,
                const ElementIndex<VolumeDim>& b) noexcept {
  return not(a == b);
}

template <size_t VolumeDim>
size_t hash_value(const ElementIndex<VolumeDim>& index) noexcept {
  // ElementIndex is used as an opaque array of bytes by Charm, so we
  // treat it that way as well.
  // clang-tidy: do not use reinterpret_cast
  const auto bytes = reinterpret_cast<const char*>(&index);  // NOLINT
  // clang-tidy: do not use pointer arithmetic
  return boost::hash_range(bytes, bytes + sizeof(index));  // NOLINT
}

template <size_t VolumeDim>
size_t std::hash<ElementIndex<VolumeDim>>::operator()(
    const ElementIndex<VolumeDim>& x) const noexcept {
  return hash_value(x);
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& s,
                         const ElementIndex<VolumeDim>& index) noexcept {
  for (const auto si : index.segments()) {
    s << si;
  }
  return s;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                             \
  template struct ElementIndex<GET_DIM(data)>;                             \
  template bool operator==(const ElementIndex<GET_DIM(data)>& a,           \
                           const ElementIndex<GET_DIM(data)>& b) noexcept; \
  template bool operator!=(const ElementIndex<GET_DIM(data)>& a,           \
                           const ElementIndex<GET_DIM(data)>& b) noexcept; \
  template size_t hash_value(                                              \
      const ElementIndex<GET_DIM(data)>& index) noexcept;                  \
  template size_t std::hash<ElementIndex<GET_DIM(data)>>::operator()(      \
      const ElementIndex<GET_DIM(data)>& x) const noexcept;                \
  template std::ostream& operator<<(                                       \
      std::ostream& s, const ElementIndex<GET_DIM(data)>& index) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
