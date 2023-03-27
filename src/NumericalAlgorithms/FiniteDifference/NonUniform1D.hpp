// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <deque>

namespace fd {
/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Returns the weights for a 1D non-uniform finite difference stencil.
 *
 * These weights are for the Lagrange interpolation polynomial and its
 * derivatives evaluated at `times[0]`. If the number of times is not the same
 * as `StencilSize`, then an error will occur.
 *
 * The reason that `times` was chosen to be monotonically decreasing is because
 * the intended use of this function is with data that is stored in a
 * `std::deque` where the zeroth element is the "most recent" in time and then
 * later elements are further in the "past".
 *
 * \param times A monotonically *decreasing* sequence of times
 * \return A 2D array of all finite difference weights that correspond to the
 * input times. The outer dimension is the Nth derivative, and the inner
 * dimension loops over the weights for each of the times.
 *
 * \note Only stencil sizes 2, 3, and 4 are implemented right now. If you need
 * more, you'll either need to add it yourself or generalize the algorithm.
 */
template <size_t StencilSize>
std::array<std::array<double, StencilSize>, StencilSize> non_uniform_1d_weights(
    const std::deque<double>& times);
}  // namespace fd
