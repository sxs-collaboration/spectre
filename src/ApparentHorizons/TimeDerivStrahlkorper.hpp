// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <utility>

/// \cond
template <typename Frame>
class Strahlkorper;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ah {
/// \brief Compute the time derivative of a Strahlkorper from a number of
/// previous Strahlkorpers
///
/// \details Does simple 1D FD with non-uniform spacing using
/// `fd::non_uniform_1d_weights`.
/// \param previous_strahlkorpers All previous Strahlkorpers and the times they
/// are at. They are expected to have the most recent Strahlkorper in the front
/// and the Strahlkorper furthest in the past in the back of the deque.
template <typename Frame>
void time_deriv_of_strahlkorper(
    gsl::not_null<Strahlkorper<Frame>*> time_deriv,
    const std::deque<std::pair<double, ::Strahlkorper<Frame>>>&
        previous_strahlkorpers);
}  // namespace ah
