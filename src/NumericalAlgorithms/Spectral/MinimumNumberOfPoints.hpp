// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/Spectral/Limits.hpp"

namespace Spectral {
template <Basis basis, Quadrature quadrature>
constexpr size_t minimum_number_of_points = limits::min(basis, quadrature);
}  // namespace Spectral
