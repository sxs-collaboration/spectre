// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "NumericalAlgorithms/Spectral/Basis.hpp"

namespace Spectral {
/*!
 * \brief Maximum number of allowed collocation points.
 *
 * \details We choose a limit of 24 FD grid points because for DG-subcell the
 * number of points in an element is `2 * number_dg_points - 1` for cell
 * centered, and `2 * number_dg_points` for face-centered. Because there is no
 * way of generically retrieving the maximum number of grid points for a non-FD
 * basis, we need to hard-code both values here. If the number of grid points is
 * increased for the non-FD bases, it should also be increased for the FD basis.
 * Note that for good task-based parallelization 24 grid points is already a
 * fairly large number.
 */
template <Basis basis>
constexpr size_t maximum_number_of_points =
    basis == Basis::FiniteDifference ? 40 : 20;
}  // namespace Spectral
