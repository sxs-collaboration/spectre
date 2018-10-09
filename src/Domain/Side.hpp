// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines enum class Side.

#pragma once

#include <iosfwd>

/// \ingroup ComputationalDomainGroup
/// A label for the side of a manifold.
///
/// Lower and Upper are with respect to the logical coordinate whose axis is
/// normal to the side, i.e. beyond the Upper (Lower) side, the logical
/// coordinate is increasing (decreasing).
enum class Side { Lower, Upper };

/// The opposite side
constexpr inline Side opposite(const Side side) {
  return (Side::Lower == side ? Side::Upper : Side::Lower);
}

/// Output operator for a Side.
std::ostream& operator<<(std::ostream& os, const Side& side) noexcept;
