// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines enum class Side.

#pragma once

#include <cstdint>
#include <iosfwd>

namespace detail {
constexpr uint8_t side_shift = 2;
}  // namespace detail

/// \ingroup ComputationalDomainGroup
/// A label for the side of a manifold.
///
/// Lower and Upper are with respect to the logical coordinate whose axis is
/// normal to the side, i.e. beyond the Upper (Lower) side, the logical
/// coordinate is increasing (decreasing).
///
/// `Self` is used to mark when an `ElementId` does not have a direction.
///
/// Currently `Direction` assumes this enum uses no more than 2 bits.
enum class Side : uint8_t {
  Uninitialized = 0 << detail::side_shift,
  Lower = 1 << detail::side_shift,
  Upper = 2 << detail::side_shift,
  Self = 3 << detail::side_shift
};

/// The opposite side
constexpr inline Side opposite(const Side side) {
  return (Side::Lower == side ? Side::Upper : Side::Lower);
}

/// Output operator for a Side.
std::ostream& operator<<(std::ostream& os, const Side& side);
