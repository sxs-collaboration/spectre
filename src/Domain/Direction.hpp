// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Direction.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <iosfwd>

#include "Domain/Side.hpp"

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
  enum class Axis;

  /// Construct by specifying an Axis and a Side.
  Direction(Axis axis, Side side) noexcept : axis_(axis), side_(side) {}

  /// Construct by specifying a dimension and a Side.
  Direction<VolumeDim>(size_t dimension, Side side) noexcept;

  /// Default constructor for Charm++ serialization.
  Direction() noexcept;

  /// The dimension of the Direction
  size_t dimension() const noexcept { return static_cast<size_t>(axis_); }

  /// The Axis of the Direction
  Axis axis() const noexcept { return axis_; }

  /// The side of the Direction
  Side side() const noexcept { return side_; }

  /// The sign for the normal to the Side.
  double sign() const noexcept { return (Side::Lower == side_ ? -1.0 : 1.0); }

  /// The opposite Direction.
  Direction<VolumeDim> opposite() const noexcept;

  // An array of all logical Directions for a given dimensionality.
  static const std::array<Direction<VolumeDim>, 2 * VolumeDim>&
  all_directions() noexcept;

  // {@
  /// Helper functions for creating specific Directions.
  /// These are labeled by the logical-coordinate names (Xi,Eta,Zeta).
  // Note: these are functions because they contain static_assert.
  static Direction<VolumeDim> lower_xi() noexcept;
  static Direction<VolumeDim> upper_xi() noexcept;
  static Direction<VolumeDim> lower_eta() noexcept;
  static Direction<VolumeDim> upper_eta() noexcept;
  static Direction<VolumeDim> lower_zeta() noexcept;
  static Direction<VolumeDim> upper_zeta() noexcept;
  // @}

  /// Serialization for Charm++
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  Axis axis_{Axis::Xi};
  Side side_{Side::Lower};
};

/// Output operator for a Direction.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Direction<VolumeDim>& direction) noexcept;

//##############################################################################
// INLINE DEFINITIONS
//##############################################################################

// clang-tidy: redundant declaration false positive. Needs to be here because of
// the Axis enum, otherwise won't compile.
template <size_t VolumeDim>
Direction<VolumeDim>::Direction() noexcept = default;  // NOLINT

// Needed in order to address warning; ignorning -Wpedantic is needed.
// Bug 61491
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61491
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
template <>
enum class Direction<1>::Axis{Xi = 0};

template <>
enum class Direction<2>::Axis{Xi = 0, Eta = 1};

template <>
enum class Direction<3>::Axis{Xi = 0, Eta = 1, Zeta = 2};
#pragma GCC diagnostic pop

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::lower_xi() noexcept {
  return Direction(Direction<VolumeDim>::Axis::Xi, Side::Lower);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::upper_xi() noexcept {
  return Direction(Direction<VolumeDim>::Axis::Xi, Side::Upper);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::lower_eta() noexcept {
  static_assert(VolumeDim == 2 or VolumeDim == 3, "VolumeDim must be 2 or 3.");
  return Direction(Direction<VolumeDim>::Axis::Eta, Side::Lower);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::upper_eta() noexcept {
  static_assert(VolumeDim == 2 or VolumeDim == 3, "VolumeDim must be 2 or 3.");
  return Direction(Direction<VolumeDim>::Axis::Eta, Side::Upper);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::lower_zeta() noexcept {
  static_assert(VolumeDim == 3, "VolumeDim must be 3.");
  return Direction(Direction<VolumeDim>::Axis::Zeta, Side::Lower);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::upper_zeta() noexcept {
  static_assert(VolumeDim == 3, "VolumeDim must be 3.");
  return Direction(Direction<VolumeDim>::Axis::Zeta, Side::Upper);
}

template <size_t VolumeDim>
inline Direction<VolumeDim> Direction<VolumeDim>::opposite() const noexcept {
  return Direction<VolumeDim>(axis_, ::opposite(side_));
}

/// \cond NEVER
template <>
inline const std::array<Direction<1>, 2>& Direction<1>::all_directions()
    noexcept {
  const static auto directions = std::array<Direction<1>, 2>{
      {Direction<1>::upper_xi(), Direction<1>::lower_xi()}};
  return directions;
}

template <>
inline const std::array<Direction<2>, 4>& Direction<2>::all_directions()
    noexcept {
  const static auto directions = std::array<Direction<2>, 4>{
      {Direction<2>::upper_xi(), Direction<2>::lower_xi(),
       Direction<2>::upper_eta(), Direction<2>::lower_eta()}};
  return directions;
}

template <>
inline const std::array<Direction<3>, 6>& Direction<3>::all_directions()
    noexcept {
  const static auto directions = std::array<Direction<3>, 6>{
      {Direction<3>::upper_xi(), Direction<3>::lower_xi(),
       Direction<3>::upper_eta(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta(), Direction<3>::lower_zeta()}};
  return directions;
}
/// \endcond

template <size_t VolumeDim>
bool operator==(const Direction<VolumeDim>& lhs,
                const Direction<VolumeDim>& rhs) noexcept {
  return lhs.dimension() == rhs.dimension() and lhs.sign() == rhs.sign();
}

template <size_t VolumeDim>
bool operator!=(const Direction<VolumeDim>& lhs,
                const Direction<VolumeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

template <size_t VolumeDim>
size_t hash_value(const Direction<VolumeDim>& d) noexcept {
  return std::hash<size_t>{}(d.dimension()) xor std::hash<double>{}(d.sign());
}

namespace std {
template <size_t VolumeDim>
struct hash<Direction<VolumeDim>> {
  size_t operator()(const Direction<VolumeDim>& d) const noexcept {
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

  size_t operator()(const Direction<Dim>& t) noexcept {
    return 2 * t.dimension() + (t.side() == Side::Upper ? 1 : 0);
  }
};
