// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ElementId.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <regex>
#include <string>
#include <type_traits>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/StdHelpers/Bit.hpp"

// The `static_assert`s verify that `ElementId` satisfies the constraints
// imposed by Charm++ to make `ElementId` able to act as an index into Charm++'s
// arrays. These constraints are:
// - `ElementId` must satisfy `std::is_standard_layout` and `std::is_trivial`
// - `ElementId` is the size of two `int`s`
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

// IMPLEMENTATION DETAILS:
// We use a compact representation of a SegmentId as a uint16_t.
// This allows 16 refinement levels from the coarsest level 0 thorugh 15
// Labeling the 16 bits of the uint16_t from right (0) to left (15), the
// position of the bit of the leading 1 (also known as the most significant bit)
// determines the refinement level of the SegmentId, while masking that bit
// (i.e. setting it to zero) of the uint16_t will determine the index of the
// SegmentId.
// We reserve 0 as representing an undefined SegmentId
// Thus using the string representation of a SegmentId as LlIi with l and i
// as the refinement level and index respectively we have:
// 0000000000000000 undefined
// 0000000000000001 L0I0
// 0000000000000010 L1I0
// 0000000000000011 L1I1
// 0000000000000100 L2I0
// ...
// 1111111111111111 L15I32767
namespace {
uint16_t make_compact_segment_id(const SegmentId& segment_id) {
  return (uint16_t{1} << segment_id.refinement_level()) +
         static_cast<uint16_t>(segment_id.index());
}

size_t get_refinement_level(const uint16_t compact_segment_id) {
  ASSERT(compact_segment_id > 0, "Undefined compact segment id");
  return 15_st - static_cast<size_t>(std::countl_zero(compact_segment_id));
}

size_t get_index(const uint16_t compact_segment_id) {
  ASSERT(compact_segment_id > 0, "Undefined compact segment id");
  return static_cast<size_t>(compact_segment_id -
                             std::bit_floor(compact_segment_id));
}

SegmentId make_segment_id(const uint16_t compact_segment_id) {
  ASSERT(compact_segment_id > 0, "Undefined compact segment id");
  return SegmentId{get_refinement_level(compact_segment_id),
                   get_index(compact_segment_id)};
}

bool is_on_lower_block_boundary(const uint16_t compact_segment_id) {
  ASSERT(compact_segment_id > 0, "Undefined compact segment id");
  // true if index is zero, which means only a single bit is a one
  return std::has_single_bit(compact_segment_id);
}

bool is_on_upper_block_boundary(const uint16_t compact_segment_id) {
  ASSERT(compact_segment_id > 0, "Undefined compact segment id");
  // true if all bits after the leading one are also one
  return 16 == (std::countl_zero(compact_segment_id) +
                std::countr_one(compact_segment_id));
}
}  // namespace

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const uint8_t block_id,
                                const uint8_t grid_index,
                                const uint8_t direction,
                                const uint16_t compact_segment_id_xi,
                                const uint16_t compact_segment_id_eta,
                                const uint16_t compact_segment_id_zeta)
    : block_id_(block_id),
      grid_index_(grid_index),
      direction_(direction),
      compact_segment_id_xi_(compact_segment_id_xi),
      compact_segment_id_eta_(compact_segment_id_eta),
      compact_segment_id_zeta_(compact_segment_id_zeta) {}

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const size_t block_id, const size_t grid_index)
    : block_id_(static_cast<uint8_t>(block_id)),
      grid_index_(static_cast<uint8_t>(grid_index)),
      direction_(Direction<VolumeDim>::self().bits()),
      compact_segment_id_xi_(uint16_t{1}),
      compact_segment_id_eta_(VolumeDim > 1 ? uint16_t{1} : uint16_t{0}),
      compact_segment_id_zeta_(VolumeDim > 2 ? uint16_t{1} : uint16_t{0}) {
  ASSERT(block_id < two_to_the(block_id_bits),
         "Block id out of bounds: " << block_id << "\nMaximum value is: "
                                    << two_to_the(block_id_bits) - 1);
  ASSERT(grid_index < two_to_the(grid_index_bits),
         "Grid index out of bounds: " << grid_index << "\nMaximum value is: "
                                      << two_to_the(grid_index_bits) - 1);
}

template <>
ElementId<1>::ElementId(const size_t block_id,
                        const std::array<SegmentId, 1>& segment_ids,
                        const size_t grid_index)
    : block_id_(static_cast<uint8_t>(block_id)),
      grid_index_(static_cast<uint8_t>(grid_index)),
      direction_(Direction<3>::self().bits()),
      compact_segment_id_xi_(make_compact_segment_id(segment_ids[0])),
      compact_segment_id_eta_(uint16_t{0}),
      compact_segment_id_zeta_(uint16_t{0}) {
  ASSERT(block_id < two_to_the(block_id_bits),
         "Block id out of bounds: " << block_id << "\nMaximum value is: "
                                    << two_to_the(block_id_bits) - 1);
  ASSERT(grid_index < two_to_the(grid_index_bits),
         "Grid index out of bounds: " << grid_index << "\nMaximum value is: "
                                      << two_to_the(grid_index_bits) - 1);
}

template <>
ElementId<2>::ElementId(const size_t block_id,
                        const std::array<SegmentId, 2>& segment_ids,
                        const size_t grid_index)
    : block_id_(static_cast<uint8_t>(block_id)),
      grid_index_(static_cast<uint8_t>(grid_index)),
      direction_(Direction<3>::self().bits()),
      compact_segment_id_xi_(make_compact_segment_id(segment_ids[0])),
      compact_segment_id_eta_(make_compact_segment_id(segment_ids[1])),
      compact_segment_id_zeta_(uint16_t{0}) {
  ASSERT(block_id < two_to_the(block_id_bits),
         "Block id out of bounds: " << block_id << "\nMaximum value is: "
                                    << two_to_the(block_id_bits) - 1);
  ASSERT(grid_index < two_to_the(grid_index_bits),
         "Grid index out of bounds: " << grid_index << "\nMaximum value is: "
                                      << two_to_the(grid_index_bits) - 1);
}

template <>
ElementId<3>::ElementId(const size_t block_id,
                        const std::array<SegmentId, 3>& segment_ids,
                        const size_t grid_index)
    : block_id_(static_cast<uint8_t>(block_id)),
      grid_index_(static_cast<uint8_t>(grid_index)),
      direction_(Direction<3>::self().bits()),
      compact_segment_id_xi_(make_compact_segment_id(segment_ids[0])),
      compact_segment_id_eta_(make_compact_segment_id(segment_ids[1])),
      compact_segment_id_zeta_(make_compact_segment_id(segment_ids[2])) {
  ASSERT(block_id < two_to_the(block_id_bits),
         "Block id out of bounds: " << block_id << "\nMaximum value is: "
                                    << two_to_the(block_id_bits) - 1);
  ASSERT(grid_index < two_to_the(grid_index_bits),
         "Grid index out of bounds: " << grid_index << "\nMaximum value is: "
                                      << two_to_the(grid_index_bits) - 1);
}

template <size_t VolumeDim>
ElementId<VolumeDim>::ElementId(const Direction<VolumeDim>& direction,
                                const ElementId<VolumeDim>& element_id)
    : block_id_(element_id.block_id_),
      grid_index_(element_id.grid_index_),
      direction_(direction.bits()),
      compact_segment_id_xi_(element_id.compact_segment_id_xi_),
      compact_segment_id_eta_(element_id.compact_segment_id_eta_),
      compact_segment_id_zeta_(element_id.compact_segment_id_zeta_) {}

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
  block_id_ = to_size_t(match[1]);
  const auto make_compact_segment_id = [](const size_t refinement_level,
                                          const size_t index,
                                          const std::string& name) {
    ASSERT(refinement_level <= max_refinement_level,
           "Refinement level '"
               << refinement_level << "' out of bounds for element ID '" << name
               << "'. Maximum value is: " << max_refinement_level);
    ASSERT(index < two_to_the(refinement_level),
           "Index '" << index << "' out of bounds for element ID '" << name
                     << "'. Maximum value is: "
                     << two_to_the(refinement_level) - 1);
    return (uint16_t{1} << refinement_level) + static_cast<uint16_t>(index);
  };

  compact_segment_id_xi_ = make_compact_segment_id(
      to_size_t(match[2]), to_size_t(match[3]), grid_name);
  if constexpr (VolumeDim > 1) {
    compact_segment_id_eta_ = make_compact_segment_id(
        to_size_t(match[4]), to_size_t(match[5]), grid_name);
  } else {
    compact_segment_id_eta_ = 0;
  }
  if constexpr (VolumeDim > 2) {
    compact_segment_id_zeta_ = make_compact_segment_id(
        to_size_t(match[6]), to_size_t(match[7]), grid_name);
  } else {
    compact_segment_id_zeta_ = 0;
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
  ASSERT(dim < VolumeDim,
         "Dimension must be smaller than " << VolumeDim << ", but is: " << dim);
  ElementId<VolumeDim> result = this->without_direction();
  switch (dim) {
    case 0:
      ASSERT(get_refinement_level(result.compact_segment_id_xi_) !=
                 max_refinement_level,
             "Cannot get child of element on max refinement level");
      result.compact_segment_id_xi_ = result.compact_segment_id_xi_ << 1;
      if (side == Side::Upper) {
        ++result.compact_segment_id_xi_;
      }
      return result;
    case 1:
      ASSERT(get_refinement_level(result.compact_segment_id_eta_) !=
                 max_refinement_level,
             "Cannot get child of element on max refinement level");
      result.compact_segment_id_eta_ = result.compact_segment_id_eta_ << 1;
      if (side == Side::Upper) {
        ++result.compact_segment_id_eta_;
      }
      return result;
    case 2:
      ASSERT(get_refinement_level(result.compact_segment_id_zeta_) !=
                 max_refinement_level,
             "Cannot get child of element on max refinement level");
      result.compact_segment_id_zeta_ = result.compact_segment_id_zeta_ << 1;
      if (side == Side::Upper) {
        ++result.compact_segment_id_zeta_;
      }
      return result;
    default:
      ERROR("Invalid dimension: " << dim);
  }
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::id_of_parent(
    const size_t dim) const {
  ASSERT(dim < VolumeDim,
         "Dimension must be smaller than " << VolumeDim << ", but is: " << dim);
  ElementId<VolumeDim> result = this->without_direction();
  switch (dim) {
    case 0:
      ASSERT(get_refinement_level(result.compact_segment_id_xi_) != 0,
             "Cannot get parent of element on refinement level 0");
      result.compact_segment_id_xi_ = result.compact_segment_id_xi_ >> 1;
      return result;
    case 1:
      ASSERT(get_refinement_level(result.compact_segment_id_xi_) != 0,
             "Cannot get parent of element on refinement level 0");
      result.compact_segment_id_eta_ = result.compact_segment_id_eta_ >> 1;
      return result;
    case 2:
      ASSERT(get_refinement_level(result.compact_segment_id_xi_) != 0,
             "Cannot get parent of element on refinement level 0");
      result.compact_segment_id_zeta_ = result.compact_segment_id_zeta_ >> 1;
      return result;
    default:
      ERROR("Invalid dimension: " << dim);
  }
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
    return {{get_refinement_level(compact_segment_id_xi_)}};
  } else if constexpr (VolumeDim == 2) {
    return {{get_refinement_level(compact_segment_id_xi_),
             get_refinement_level(compact_segment_id_eta_)}};
  } else if constexpr (VolumeDim == 3) {
    return {{get_refinement_level(compact_segment_id_xi_),
             get_refinement_level(compact_segment_id_eta_),
             get_refinement_level(compact_segment_id_zeta_)}};
  }
}

template <size_t VolumeDim>
std::array<SegmentId, VolumeDim> ElementId<VolumeDim>::segment_ids() const {
  if constexpr (VolumeDim == 1) {
    return {{make_segment_id(compact_segment_id_xi_)}};
  } else if constexpr (VolumeDim == 2) {
    return {{make_segment_id(compact_segment_id_xi_),
             make_segment_id(compact_segment_id_eta_)}};
  } else if constexpr (VolumeDim == 3) {
    return {{make_segment_id(compact_segment_id_xi_),
             make_segment_id(compact_segment_id_eta_),
             make_segment_id(compact_segment_id_zeta_)}};
  }
}

template <size_t VolumeDim>
SegmentId ElementId<VolumeDim>::segment_id(const size_t dim) const {
  ASSERT(dim < VolumeDim,
         "Dimension must be smaller than " << VolumeDim << ", but is: " << dim);
  switch (dim) {
    case 0:
      return make_segment_id(compact_segment_id_xi_);
    case 1:
      return make_segment_id(compact_segment_id_eta_);
    case 2:
      return make_segment_id(compact_segment_id_zeta_);
    default:
      ERROR("Invalid dimension: " << dim);
  }
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::external_boundary_id() {
  // In order to distinguish this from an uninitialized ElementId, we use the
  // maximum possible value that can be stored in `block_id_bits`
  static_assert(ElementId::block_id_bits == 8);
  return ElementId{255, 0, 0, 0, 0, 0};
}

template <size_t VolumeDim>
ElementId<VolumeDim> ElementId<VolumeDim>::without_direction() const {
  ElementId result = *this;
  result.direction_ = Direction<VolumeDim>::self().bits();
  return result;
}

template <size_t VolumeDim>
size_t ElementId<VolumeDim>::number_of_block_boundaries() const {
  return (is_on_lower_block_boundary(compact_segment_id_xi_) ? 1_st : 0_st) +
         (is_on_upper_block_boundary(compact_segment_id_xi_) ? 1_st : 0_st) +
         (VolumeDim > 1
              ? (is_on_lower_block_boundary(compact_segment_id_eta_) ? 1_st
                                                                     : 0_st) +
                    (is_on_upper_block_boundary(compact_segment_id_eta_) ? 1_st
                                                                         : 0_st)
              : 0_st) +
         (VolumeDim > 2
              ? (is_on_lower_block_boundary(compact_segment_id_zeta_) ? 1_st
                                                                      : 0_st) +
                    (is_on_upper_block_boundary(compact_segment_id_zeta_)
                         ? 1_st
                         : 0_st)
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
  if (lhs.grid_index_ != rhs.grid_index_) {
    return lhs.grid_index_ < rhs.grid_index_;
  }
  if (lhs.block_id_ != rhs.block_id_) {
    return lhs.block_id_ < rhs.block_id_;
  }
  if (lhs.compact_segment_id_xi_ != rhs.compact_segment_id_xi_) {
    return lhs.compact_segment_id_xi_ < rhs.compact_segment_id_xi_;
  }
  if constexpr (VolumeDim > 1) {
    if (lhs.compact_segment_id_eta_ != rhs.compact_segment_id_eta_) {
      return lhs.compact_segment_id_eta_ < rhs.compact_segment_id_eta_;
    }
  }
  if constexpr (VolumeDim > 2) {
    if (lhs.compact_segment_id_zeta_ != rhs.compact_segment_id_zeta_) {
      return lhs.compact_segment_id_zeta_ < rhs.compact_segment_id_zeta_;
    }
  }
  return false;
}

template <size_t Dim>
bool is_zeroth_element(const ElementId<Dim>& id,
                       const std::optional<size_t>& grid_index) {
  if (id.block_id_ != 0) {
    return false;
  }
  if (not is_on_lower_block_boundary(id.compact_segment_id_xi_)) {
    return false;
  }
  if (Dim > 1 and not is_on_lower_block_boundary(id.compact_segment_id_eta_)) {
    return false;
  }
  if (Dim > 2 and not is_on_lower_block_boundary(id.compact_segment_id_zeta_)) {
    return false;
  }
  if (grid_index.has_value()) {
    return id.grid_index_ == grid_index.value();
  }
  return true;
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

// NOLINTNEXTLINE(cert-dcl58-cpp)
namespace std {
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
