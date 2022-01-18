// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "ApparentHorizons/Strahlkorper.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

// Create a strahlkorper with a Im(Y11) dependence, with
// a given average radius and a given center.
Strahlkorper<Frame::Inertial> create_strahlkorper_y11(
    const double y11_amplitude, const double radius,
    const std::array<double, 3>& center);
