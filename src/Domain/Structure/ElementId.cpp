// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ElementId.hpp"

#include <boost/functional/hash.hpp>
#include <exception>
#include <limits>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <regex>
#include <string>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/StdHelpers.hpp"

// The `static_assert`s verify that `ElementId` satisfies the constraints
// imposed by Charm++ to make `ElementId` able to act as an index into Charm++'s
// arrays. These constraints are:
// - `ElementId` must satisfy `std::is_standard_layout` and `std::is_trivial`
// - `ElementId` must not be larger than the size of three `int`s, i.e.
//   `sizeof(ElementId) <= 3 * sizeof(int)`
static_assert(std::is_standard_layout_v<ElementId<1>> and
              std::is_trivial_v<ElementId<1>>);
static_assert(std::is_standard_layout_v<ElementId<2>> and
              std::is_trivial_v<ElementId<2>>);
static_assert(std::is_standard_layout_v<ElementId<3>> and
              std::is_trivial_v<ElementId<3>>);

static_assert(sizeof(ElementId<1>) == 2 * sizeof(int),
              "Wrong size for ElementId<1>");
static_assert(sizeof(ElementId<2>) == 2 * sizeof(int),
              "Wrong size for ElementId<2>");
static_assert(sizeof(ElementId<3>) == 2 * sizeof(int),
              "Wrong size for ElementId<3>");

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const size_t block_id, const size_t grid_index)
    : block_id_(block_id),
      grid_index_(grid_index),
      direction_{Direction<VolumeDim>::self().bits()},
      index_xi_{0},
      refinement_level_xi_{0},
      index_eta_{0},
      refinement_level_eta_{0},
      index_zeta_{0},
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
    : block_id_(block_id),
      grid_index_(grid_index),
      direction_{Direction<VolumeDim>::self().bits()} {
  ASSERT(block_id < two_to_the(block_id_bits),
         "Block id out of bounds: " << block_id << "\nMaximum value is: "
                                    << two_to_the(block_id_bits) - 1);
  ASSERT(grid_index < two_to_the(grid_index_bits),
         "Grid index out of bounds: " << grid_index << "\nMaximum value is: "
                                      << two_to_the(grid_index_bits) - 1);
  const auto check_refinement_level = [](const size_t refinement_level) {
    ASSERT(refinement_level <= max_refinement_level,
           "Refinement level out of bounds: " << refinement_level
                                              << "\nMaximum value is: "
                                              << max_refinement_level);
    return refinement_level;
  };
  index_xi_ = segment_ids[0].index();
  refinement_level_xi_ =
      check_refinement_level(segment_ids[0].refinement_level());
  if constexpr (VolumeDim > 1) {
    index_eta_ = segment_ids[1].index();
    refinement_level_eta_ =
        check_refinement_level(segment_ids[1].refinement_level());
  } else {
    index_eta_ = 0;
    refinement_level_eta_ = 0;
  }
  if constexpr (VolumeDim > 2) {
    index_zeta_ = segment_ids[2].index();
    refinement_level_zeta_ =
        check_refinement_level(segment_ids[2].refinement_level());
  } else {
    index_zeta_ = 0;
    refinement_level_zeta_ = 0;
  }
}

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const Direction<VolumeDim>& direction,
                                const ElementId<VolumeDim>& element_id)
    : block_id_(element_id.block_id_),
      grid_index_(element_id.grid_index_),
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      direction_(direction.bits()),
      index_xi_{element_id.index_xi_},
      refinement_level_xi_{element_id.refinement_level_xi_},
      index_eta_{element_id.index_eta_},
      refinement_level_eta_{element_id.refinement_level_eta_},
      index_zeta_{element_id.index_zeta_},
      refinement_level_zeta_{element_id.refinement_level_zeta_} {}

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const std::string& grid_name)
    : direction_{Direction<VolumeDim>::self().bits()} {
  std::string pattern_str = "\\[B([0-9]+),\\(";
  for (size_t d = 0; d < VolumeDim; ++d) {
    pattern_str.append("L([0-9]+)I([0-9]+)");
    if (d < VolumeDim - 1) {
      pattern_str.append(",");
    }
  }
  pattern_str.append("\\)(,G([0-9]+))?\\]");
  const std::regex pattern(pattern_str);
  std::smatch match;
  std::regex_match(grid_name, match, pattern);
  if (match.empty()) {
    throw std::invalid_argument{"Invalid grid name '" + grid_name +
                                "' does not match the pattern '" + pattern_str +
                                "'."};
  }
  const auto to_size_t = [](const std::ssub_match& s) {
    return static_cast<size_t>(std::stoi(s.str()));
  };
  const auto check_refinement_level =
      [&grid_name](const size_t refinement_level) {
        ASSERT(refinement_level <= ElementId<VolumeDim>::max_refinement_level,
               "Refinement level '"
                   << refinement_level << "' out of bounds for element ID '"
                   << grid_name << "'. Maximum value is: "
                   << ElementId<VolumeDim>::max_refinement_level);
        return refinement_level;
      };
  block_id_ = to_size_t(match[1]);
  refinement_level_xi_ = check_refinement_level(to_size_t(match[2]));
  index_xi_ = to_size_t(match[3]);
  if constexpr (VolumeDim > 1) {
    refinement_level_eta_ = check_refinement_level(to_size_t(match[4]));
    index_eta_ = to_size_t(match[5]);
  } else {
    refinement_level_eta_ = 0;
    index_eta_ = 0;
  }
  if constexpr (VolumeDim > 2) {
    refinement_level_zeta_ = check_refinement_level(to_size_t(match[6]));
    index_zeta_ = to_size_t(match[7]);
  } else {
    refinement_level_zeta_ = 0;
    index_zeta_ = 0;
  }
  if (match[1                // Full matched string
            + 1              // Block ID
            + 2 * VolumeDim  // Segment IDs
            + 1              // Optional grid_index subexpression
  ]
          .matched) {
    grid_index_ = to_size_t(match[2 * VolumeDim + 3]);
  } else {
    grid_index_ = 0;
  }
  ASSERT(block_id_ < two_to_the(block_id_bits),
         "Block id out of bounds: " << block_id_ << "\nMaximum value is: "
                                    << two_to_the(block_id_bits) - 1);
  ASSERT(grid_index_ < two_to_the(grid_index_bits),
         "Grid index out of bounds: " << grid_index_ << "\nMaximum value is: "
                                      << two_to_the(grid_index_bits) - 1);
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
Direction<VolumeDim> ElementId<VolumeDim>::direction() const {
  return Direction<VolumeDim>{static_cast<typename Direction<VolumeDim>::Axis>(
                                  direction_ bitand 0b0011),
                              static_cast<Side>(direction_ bitand 0b1100)};
}

template <size_t VolumeDim>
std::array<size_t, VolumeDim> ElementId<VolumeDim>::refinement_levels() const {
  if constexpr (VolumeDim == 1) {
    return {{refinement_level_xi_}};
  } else if constexpr (VolumeDim == 2) {
    return {{refinement_level_xi_, refinement_level_eta_}};
  } else if constexpr (VolumeDim == 3) {
    return {
        {refinement_level_xi_, refinement_level_eta_, refinement_level_zeta_}};
  }
}

template <size_t VolumeDim>
std::array<SegmentId, VolumeDim> ElementId<VolumeDim>::segment_ids() const {
  if constexpr (VolumeDim == 1) {
    return {{SegmentId{refinement_level_xi_, index_xi_}}};
  } else if constexpr (VolumeDim == 2) {
    return {{SegmentId{refinement_level_xi_, index_xi_},
             SegmentId{refinement_level_eta_, index_eta_}}};
  } else if constexpr (VolumeDim == 3) {
    return {{SegmentId{refinement_level_xi_, index_xi_},
             SegmentId{refinement_level_eta_, index_eta_},
             SegmentId{refinement_level_zeta_, index_zeta_}}};
  }
}

template <size_t VolumeDim>
SegmentId ElementId<VolumeDim>::segment_id(const size_t dim) const {
  ASSERT(dim < VolumeDim,
         "Dimension must be smaller than " << VolumeDim << ", but is: " << dim);
  switch (dim) {
    case 0:
      return {refinement_level_xi_, index_xi_};
    case 1:
      return {refinement_level_eta_, index_eta_};
    case 2:
      return {refinement_level_zeta_, index_zeta_};
    default:
      ERROR("Invalid dimension: " << dim);
  }
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
ElementId<VolumeDim> ElementId<VolumeDim>::without_direction() const {
  ElementId result = *this;
  result.direction_ = Direction<VolumeDim>::self().bits();
  return result;
}

template <size_t VolumeDim>
size_t ElementId<VolumeDim>::number_of_block_boundaries() const {
  return (index_xi_ == 0 ? (refinement_level_xi_ == 0 ? 2 : 1)
          : (index_xi_ ==
             two_to_the(static_cast<unsigned short>(refinement_level_xi_)) - 1)
              ? 1_st
              : 0_st) +
         (VolumeDim > 1
              ? (index_eta_ == 0
                     ? (refinement_level_eta_ == 0 ? 2 : 1)
                     : ((index_eta_ == two_to_the(static_cast<unsigned short>(
                                           refinement_level_eta_)) -
                                           1)
                            ? 1_st
                            : 0_st))
              : 0_st) +
         (VolumeDim > 2
              ? (index_zeta_ == 0
                     ? (refinement_level_zeta_ == 0 ? 2 : 1)
                     : ((index_zeta_ == two_to_the(static_cast<unsigned short>(
                                            refinement_level_zeta_)) -
                                            1)
                            ? 1_st
                            : 0_st))
              : 0_st);
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

template <size_t Dim>
bool is_zeroth_element(const ElementId<Dim>& id,
                       const std::optional<size_t>& grid_index) {
  const bool base_checks =
      id.block_id() == 0 and
      alg::accumulate(id.segment_ids(), 0_st,
                      [](const size_t current, const auto& segment_id) {
                        return current + segment_id.index();
                      }) == 0;
  return grid_index.has_value()
             ? base_checks and id.grid_index() == grid_index.value()
             : base_checks;
}

template <size_t Dim>
bool is_zeroth_element(const ElementId<Dim>& id) {
  return is_zeroth_element(id, std::nullopt);
}

// LCOV_EXCL_START
template <size_t VolumeDim>
size_t hash_value(const ElementId<VolumeDim>& id) {
  // ElementId is used as an opaque array of bytes by Charm, so we
  // treat it that way as well. However, since only the lowest 64 bits are
  // used, the hash is trivial: just extract the lowest 64 bits.
  // clang-tidy: do not use reinterpret_cast
  return (*reinterpret_cast<const uint64_t*>(&id)) bitand  // NOLINT
         (compl ElementId<VolumeDim>::direction_mask);
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

#define INSTANTIATION(r, data)                                              \
  template class ElementId<GET_DIM(data)>;                                  \
  template std::ostream& operator<<(std::ostream&,                          \
                                    const ElementId<GET_DIM(data)>&);       \
  template bool operator<(const ElementId<GET_DIM(data)>& lhs,              \
                          const ElementId<GET_DIM(data)>& rhs);             \
  template bool is_zeroth_element(const ElementId<GET_DIM(data)>& id,       \
                                  const std::optional<size_t>& grid_index); \
  template bool is_zeroth_element(const ElementId<GET_DIM(data)>& id);      \
  template size_t hash_value(const ElementId<GET_DIM(data)>&);              \
  namespace std { /* NOLINT */                                              \
  template struct hash<ElementId<GET_DIM(data)>>;                           \
  }

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
