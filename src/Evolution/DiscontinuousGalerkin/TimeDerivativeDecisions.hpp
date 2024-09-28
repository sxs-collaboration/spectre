// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace evolution::dg {
/*!
 * \brief Runtime control over time derivative work done.
 *
 * - `compute_flux_divergence`: if `true` then we compute and add the flux
     divergence to the volume time derivative. Set to `false` to elide work
     where you know the solution is spatially constant.
 */
template <size_t Dim>
struct TimeDerivativeDecisions {
  bool compute_flux_divergence = true;
};
}  // namespace evolution::dg
