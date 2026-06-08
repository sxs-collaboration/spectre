// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {

/*!
 * \brief Compute the time derivatives of the photon geodesic equation
 * subject to an algebraic speed-of-light constraint.
 *
 * \details
 * This function integrates the Hamiltonian equations of motion for a photon.
 * By strictly enforcing the null constraint $g_{\mu\nu} p^\mu p^\nu = 0$,
 * the time derivatives of the spatial position $x^i$ and the spatial
 * momentum $p_i$ are given by:
 * \frac{dx^i}{dt} = \gamma^{ij} \frac{p_j}{p^0} - \beta^i
 * \frac{dp_i}{dt} = dp_i / dt = - \alpha (\partial_i \alpha) p^0
 *                      + p_k \partial_i \beta^k
 *                      + \frac{1}{2 p^0} p^m p^n \partial_i \gamma_{mn}
 * where $\alpha$ is the lapse, $\beta^i$ is the shift, $\gamma^{ij}$
 * is the inverse spatial metric, and $p^0$ is determined algebraically from
 * the local spacetime geometry to prevent numerical constraint-violating
 * drift.
 *
 * \param dt_x Output: Time derivatives of the spatial position.
 * \param dt_pi Output: Time derivatives of the spatial momentum.
 * \param current_p0 Output: Current $p^0$ computed from the algebraic null
 * constraint.
 * \param current_dt_lnp0 Output: Time derivative of $\ln(p^0)$.
 * \param x Current spatial coordinates of the particle.
 * \param pi Current spatial momentum of the particle.
 * \param lapse The background spacetime lapse $\alpha$.
 * \param deriv_lapse The spatial derivative of the lapse $\partial_i \alpha$.
 * \param shift The background spacetime shift $\beta^i$.
 * \param deriv_shift The spatial derivative of the shift $\partial_i \beta^j$.
 * \param inv_spatial_metric The inverse spatial metric $\gamma^{ij}$.
 * \param deriv_spatial_metric The spatial derivative of the spatial metric
 * $\partial_k \gamma_{ij}$.
 * \param extrinsic_curvature The extrinsic curvature $K_{ij}$.
 */
template <typename DataType, size_t Dim, typename Frame>
void photon_geodesic_equation_with_constraint(
    gsl::not_null<tnsr::I<DataType, Dim, Frame>*> dt_x,
    gsl::not_null<tnsr::i<DataType, Dim, Frame>*> dt_pi,
    gsl::not_null<Scalar<DataType>*> current_p0,
    gsl::not_null<Scalar<DataType>*> current_dt_lnp0,
    const tnsr::I<DataType, Dim, Frame>& x,
    const tnsr::i<DataType, Dim, Frame>& pi,
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, Dim, Frame>& deriv_lapse,
    const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::iJ<DataType, Dim, Frame>& deriv_shift,
    const tnsr::II<DataType, Dim, Frame>& inv_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& deriv_spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature);

}  // namespace gr
