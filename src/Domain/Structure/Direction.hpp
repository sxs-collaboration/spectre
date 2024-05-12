// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Direction.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <iosfwd>

#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace PUP {
class er;
}  // namespace PUP

/// \ingroup ComputationalDomainGroup
/// A particular Side along a particular coordinate Axis.
template <size_t VolumeDim>
class Direction {
 public:
  static constexpr const size_t volume_dim = VolumeDim;

  /// The logical-coordinate names of each dimension
  enum class Axis : uint8_t;

  /// Construct by specifying an Axis and a Side.
  Direction(Axis axis, Side side);

  /// Construct by specifying a dimension and a Side.
  Direction(size_t dimension, Side side);

  /// Default constructor for Charm++ serialization.
  Direction();

  /// Get a Direction representing "self" or "no direction"
  static Direction<VolumeDim> self();

  /// The dimension of the Direction
  size_t dimension() const { return static_cast<uint8_t>(axis()); }

  /// The Axis of the Direction
  Axis axis() const { return static_cast<Axis>(bit_field_ bitand 0b0011); }

  /// The side of the Direction
  Side side() const { return static_cast<Side>(bit_field_ bitand 0b1100); }

  /// The sign for the normal to the Side.
  ///
  /// This is `+1.0` if `side() == Side::Upper` and `-1.0` if
  /// `side() == Side::Lower`, otherwise an `ASSERT` is triggered.
  double sign() const {
    ASSERT(Side::Lower == side() or Side::Upper == side(),
           "sign() is only defined for Side::Lower and Side::Upper, not "
               << side());
    return (Side::Lower == side() ? -1.0 : 1.0);
  }

  /// The opposite Direction.
  Direction<VolumeDim> opposite() const;

  // An array of all logical Directions for a given dimensionality.
  static const std::array<Direction<VolumeDim>, 2 * VolumeDim>&
  all_directions();

  /// @{
  /// Helper functions for creating specific Directions.
  /// These are labeled by the logical-coordinate names (Xi,Eta,Zeta).
  // Note: these are functions because they contain static_assert.
  static Direction<VolumeDim> lower_xi();
  static Direction<VolumeDim> upper_xi();
  static Direction<VolumeDim> lower_eta();
  static Direction<VolumeDim> upper_eta();
  static Direction<VolumeDim> lower_zeta();
  static Direction<VolumeDim> upper_zeta();
  /// @}

  uint8_t bits() const { return bit_field_; }

  /// Serialization for Charm++
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalVolumeDim>
  friend bool operator==(const Direction<LocalVolumeDim>& lhs,
                         const Direction<LocalVolumeDim>& rhs);
  template <size_t LocalVolumeDim>
  friend size_t hash_value(const Direction<LocalVolumeDim>& d);

  uint8_t bit_field_{0};
};

/// Output operator for a Direction.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Direction<VolumeDim>& direction);

//##############################################################################
// INLINE DEFINITIONS
//##############################################################################

/// \cond
// clang-tidy: redundant declaration false positive. Needs to be here because of
// the Axis enum, otherwise won't compile.
template <size_t VolumeDim>
Direction<VolumeDim>::Direction() = default;  // NOLINT

// Needed in order to address warning; ignorning -Wpedantic is needed.
// Bug 61491
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61491
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
template <>
enum class Direction<1>::Axis : uint8_t{Xi = 0};

template <>
enum class Direction<2>::Axis : uint8_t{Xi = 0, Eta = 1};

template <>
enum class Direction<3>::Axis : uint8_t{Xi = 0, Eta = 1, Zeta = 2};
#pragma GCC diagnostic pop

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::lower_xi() {
  return Direction(Direction<VolumeDim>::Axis::Xi, Side::Lower);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::upper_xi() {
  return Direction(Direction<VolumeDim>::Axis::Xi, Side::Upper);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::lower_eta() {
  static_assert(VolumeDim == 2 or VolumeDim == 3, "VolumeDim must be 2 or 3.");
  return Direction(Direction<VolumeDim>::Axis::Eta, Side::Lower);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::upper_eta() {
  static_assert(VolumeDim == 2 or VolumeDim == 3, "VolumeDim must be 2 or 3.");
  return Direction(Direction<VolumeDim>::Axis::Eta, Side::Upper);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::lower_zeta() {
  static_assert(VolumeDim == 3, "VolumeDim must be 3.");
  return Direction(Direction<VolumeDim>::Axis::Zeta, Side::Lower);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::upper_zeta() {
  static_assert(VolumeDim == 3, "VolumeDim must be 3.");
  return Direction(Direction<VolumeDim>::Axis::Zeta, Side::Upper);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::opposite() const {
  return Direction<VolumeDim>(axis(), ::opposite(side()));
}

template <>
inline const std::array<Direction<1>, 2>& Direction<1>::all_directions() {
  const static auto directions = std::array<Direction<1>, 2>{
      {Direction<1>::lower_xi(), Direction<1>::upper_xi()}};
  return directions;
}

template <>
inline const std::array<Direction<2>, 4>& Direction<2>::all_directions() {
  const static auto directions = std::array<Direction<2>, 4>{
      {Direction<2>::lower_xi(), Direction<2>::upper_xi(),
       Direction<2>::lower_eta(), Direction<2>::upper_eta()}};
  return directions;
}

template <>
inline const std::array<Direction<3>, 6>& Direction<3>::all_directions() {
  const static auto directions = std::array<Direction<3>, 6>{
      {Direction<3>::lower_xi(), Direction<3>::upper_xi(),
       Direction<3>::lower_eta(), Direction<3>::upper_eta(),
       Direction<3>::lower_zeta(), Direction<3>::upper_zeta()}};
  return directions;
}
/// \endcond

template <size_t VolumeDim>
bool operator==(const Direction<VolumeDim>& lhs,
                const Direction<VolumeDim>& rhs) {
  return lhs.bit_field_ == rhs.bit_field_;
}

template <size_t VolumeDim>
bool operator!=(const Direction<VolumeDim>& lhs,
                const Direction<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

/// Define an ordering of directions first by axis (xi, eta, zeta), then by side
/// (lower, upper). There's no particular reason for this choice of ordering.
template <size_t VolumeDim>
bool operator<(const Direction<VolumeDim>& lhs,
               const Direction<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator>(const Direction<VolumeDim>& lhs,
               const Direction<VolumeDim>& rhs) {
  return rhs < lhs;
}
template <size_t VolumeDim>
bool operator<=(const Direction<VolumeDim>& lhs,
                const Direction<VolumeDim>& rhs) {
  return !(lhs > rhs);
}
template <size_t VolumeDim>
bool operator>=(const Direction<VolumeDim>& lhs,
                const Direction<VolumeDim>& rhs) {
  return !(lhs < rhs);
}

template <size_t VolumeDim>
size_t hash_value(const Direction<VolumeDim>& d) {
  return d.bit_field_;
}

namespace std {
template <size_t VolumeDim>
struct hash<Direction<VolumeDim>> {
  size_t operator()(const Direction<VolumeDim>& d) const {
    return hash_value(d);
  }
};
}  // namespace std

/// \ingroup ComputationalDomainGroup
/// Provides a perfect hash if the size of the hash table is `2 * Dim`. To take
/// advantage of this, use the `FixedHashMap` class.
template <size_t Dim>
struct DirectionHash {
  template <size_t MaxSize>
  static constexpr bool is_perfect = MaxSize == 2 * Dim;

  size_t operator()(const Direction<Dim>& t) {
    return 2 * t.dimension() + (t.side() == Side::Upper ? 1 : 0);
  }
};
