// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ElementId.hpp"

#include <boost/functional/hash.hpp>
#include <limits>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>

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

static_assert(sizeof(ElementId<1>) == 3 * sizeof(int),
              "Wrong size for ElementId<1>");
static_assert(sizeof(ElementId<2>) == 3 * sizeof(int),
              "Wrong size for ElementId<2>");
static_assert(sizeof(ElementId<3>) == 3 * sizeof(int),
              "Wrong size for ElementId<3>");

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const size_t block_id, const size_t grid_index)
    : block_id_(block_id),
      grid_index_(grid_index),
      index_xi_{0},
      index_eta_{0},
      index_zeta_{0},
      empty_{0},
      refinement_level_xi_{0},
      refinement_level_eta_{0},
      refinement_level_zeta_{0} {
  ASSERT(block_id < two_to_the(block_id_bits),
         "Block id out of bounds: " << block_id << "\nMaximum value is: "
                                    << two_to_the(block_id_bits) - 1);
  ASSERT(grid_index < two_to_the(grid_index_bits),
         "Grid index out of bounds: " << grid_index << "\nMaximum value is: "
                                      << two_to_the(grid_index_bits) - 1);
}

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const size_t block_id,
                                std::array<SegmentId, VolumeDim> segment_ids,
                                const size_t grid_index)
    : block_id_(block_id), grid_index_(grid_index) {
  ASSERT(block_id < two_to_the(block_id_bits),
         "Block id out of bounds: " << block_id << "\nMaximum value is: "
                                    << two_to_the(block_id_bits) - 1);
  ASSERT(grid_index < two_to_the(grid_index_bits),
         "Grid index out of bounds: " << grid_index << "\nMaximum value is: "
                                      << two_to_the(grid_index_bits) - 1);
  empty_ = 0;
  index_xi_ = segment_ids[0].index();
  refinement_level_xi_ = segment_ids[0].refinement_level();
  if constexpr (VolumeDim > 1) {
    index_eta_ = segment_ids[1].index();
    refinement_level_eta_ = segment_ids[1].refinement_level();
  } else {
    index_eta_ = 0;
    refinement_level_eta_ = 0;
  }
  if constexpr (VolumeDim > 2) {
    index_zeta_ = segment_ids[2].index();
    refinement_level_zeta_ = segment_ids[2].refinement_level();
  } else {
    index_zeta_ = 0;
    refinement_level_zeta_ = 0;
  }
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::id_of_child(const size_t dim,
                                                       const Side side) const {
  std::array<SegmentId, VolumeDim> new_segment_ids = segment_ids();
  gsl::at(new_segment_ids, dim) =
      gsl::at(new_segment_ids, dim).id_of_child(side);
  return {block_id(), new_segment_ids, grid_index()};
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::id_of_parent(
    const size_t dim) const {
  std::array<SegmentId, VolumeDim> new_segment_ids = segment_ids();
  gsl::at(new_segment_ids, dim) = gsl::at(new_segment_ids, dim).id_of_parent();
  return {block_id(), new_segment_ids, grid_index()};
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::external_boundary_id() {
  // We use the maximum possible value that can be stored in `block_id_bits` and
  // the maximum refinement level to signal an external boundary. While in
  // theory this could cause a problem if we have an element at the highest
  // refinement level in the largest block (by id), in practice it is very
  // unlikely we will ever get to that large of a refinement.
  return ElementId<VolumeDim>(
      two_to_the(ElementId::block_id_bits) - 1,
      make_array<VolumeDim>(SegmentId(ElementId::max_refinement_level - 1, 0)));
}

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const ElementId<VolumeDim>& id) {
  os << "[B" << id.block_id() << ',' << id.segment_ids();
  if (id.grid_index() > 0) {
    os << ",G" << id.grid_index();
  }
  os << ']';
  return os;
}

template <size_t VolumeDim>
bool operator<(const ElementId<VolumeDim>& lhs,
               const ElementId<VolumeDim>& rhs) {
  if (lhs.grid_index() != rhs.grid_index()) {
    return lhs.grid_index() < rhs.grid_index();
  }
  if (lhs.block_id() != rhs.block_id()) {
    return lhs.block_id() < rhs.block_id();
  }
  for (size_t d = 0; d < VolumeDim; ++d) {
    if (lhs.segment_id(d).refinement_level() !=
        rhs.segment_id(d).refinement_level()) {
      return lhs.segment_id(d).refinement_level() <
             rhs.segment_id(d).refinement_level();
    }
    if (lhs.segment_id(d).index() != rhs.segment_id(d).index()) {
      return lhs.segment_id(d).index() < rhs.segment_id(d).index();
    }
  }
  return false;
}

// LCOV_EXCL_START
template <size_t VolumeDim>
size_t hash_value(const ElementId<VolumeDim>& id) {
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
    const ElementId<VolumeDim>& id) const {
  return hash_value(id);
}
}  // namespace std
// LCOV_EXCL_STOP

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                        \
  template class ElementId<GET_DIM(data)>;                            \
  template std::ostream& operator<<(std::ostream&,                    \
                                    const ElementId<GET_DIM(data)>&); \
  template bool operator<(const ElementId<GET_DIM(data)>& lhs,        \
                          const ElementId<GET_DIM(data)>& rhs);       \
  template size_t hash_value(const ElementId<GET_DIM(data)>&);        \
  namespace std { /* NOLINT */                                        \
  template struct hash<ElementId<GET_DIM(data)>>;                     \
  }

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
