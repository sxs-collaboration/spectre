// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template Direction.

#pragma once

#include <functional>
#include <iosfwd>
#include <pup.h>

#include "Domain/Side.hpp"
#include "Parallel/PupStlCpp11.hpp"

/// \ingroup ComputationalDomain
/// A particular Side along a particular coordinate Axis.
template <size_t VolumeDim>
class Direction {
 public:
  /// The logical-coordinate names of each dimension
  enum class Axis;

  /// Construct by specifying an Axis and a Side.
  Direction(Axis axis, Side side) : axis_(axis), side_(side) {}

  /// Construct by specifying a dimension and a Side.
  Direction<VolumeDim>(size_t dimension, Side side);

  /// Default constructor for Charm++ serialization.
  Direction() = default;

  /// The dimension of the Direction
  size_t dimension() const noexcept { return static_cast<size_t>(axis_); }

  /// The Axis of the Direction
  Axis axis() const noexcept { return axis_; }

  /// The side of the Direction
  Side side() const noexcept { return side_; }

  /// The sign for the normal to the Side.
  double sign() const noexcept { return (Side::Lower == side_ ? -1.0 : 1.0); }

  /// The opposite Direction.
  Direction<VolumeDim> opposite() const;

  // {@
  /// Helper functions for creating specific Directions.
  /// These are labeled by the logical-coordinate names (Xi,Eta,Zeta).
  // Note: these are functions because they contain static_assert.
  static Direction<VolumeDim> lower_xi();
  static Direction<VolumeDim> upper_xi();
  static Direction<VolumeDim> lower_eta();
  static Direction<VolumeDim> upper_eta();
  static Direction<VolumeDim> lower_zeta();
  static Direction<VolumeDim> upper_zeta();
  // @}

  /// Serialization for Charm++
  void pup(PUP::er& p);  // NOLINT

 private:
  Axis axis_{Axis::Xi};
  Side side_{Side::Lower};
};

/// Output operator for a Direction.
template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os,
                         const Direction<VolumeDim>& direction);

//##############################################################################
// INLINE DEFINITIONS
//##############################################################################

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
  return Direction<VolumeDim>(axis_, ::opposite(side_));
}

template <size_t VolumeDim>
void Direction<VolumeDim>::pup(PUP::er& p) {
  p | axis_;
  p | side_;
}

template <size_t VolumeDim>
inline bool operator==(const Direction<VolumeDim>& lhs,
                       const Direction<VolumeDim>& rhs) {
  return lhs.dimension() == rhs.dimension() and lhs.sign() == rhs.sign();
}

template <size_t VolumeDim>
inline bool operator!=(const Direction<VolumeDim>& lhs,
                       const Direction<VolumeDim>& rhs) {
  return not(lhs == rhs);
}

// These are defined so that a Direction can be used as part of a key of an
// unordered_set or unordered_map.
namespace std {
template <size_t VolumeDim>
struct hash<Direction<VolumeDim>> {
  size_t operator()(const Direction<VolumeDim>& d) const {
    return hash<size_t>{}(d.dimension()) xor hash<double>{}(d.sign());
  }
};
}  // namespace std
