// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {

/*!
 * \brief First-order formulation of the geodesic equation for null geodesics
 * that is suitable for ray tracing.
 *
 * This is the formulation of the geodesic equation that is originally used for
 * ray tracing in \cite Bohn:2014xxa (Eq. (4)) and \cite Bohn:2016afc (Eq. (6)).
 *
 * For null rays with position $x^\mu$, proper time (or affine geodesic
 * parameter) $\tau$, and four-momentum $p^\mu = dx^\mu / d\tau$, we first
 * define the momentum variable
 * \begin{equation}
 *   \Pi_i = \frac{p_i}{\alpha p^0} = \frac{p_i}{\sqrt{\gamma^{jk} p_j p_k}}
 *   \text{.}
 * \end{equation}
 * Here we work in a 3+1 decomposition of spacetime with time coordinate $t$,
 * lapse $\alpha$, shift $\beta^i$, spatial metric $\gamma_{ij}$, and
 * extrinsic curvature $K_{ij}$.
 * Then, the geodesic equation is given by
 * \begin{align}
 *   \frac{d \Pi_i}{d t} &= -\partial_i \alpha + (\Pi^j \partial_j \alpha
 *     -\alpha K_{jk}\Pi^j\Pi^k) \Pi_i) + \Pi_k \partial_i \beta^k
 *     - \frac{1}{2} \alpha \Pi_j\Pi_k \partial_i \gamma^{jk} \\
 *   \frac{d x^i}{d t} &= \alpha \Pi^i - \beta^i
 *   \text{.}
 * \end{align}
 *
 * This function also computes the evolution of the additional redshift variable
 * $\ln(\alpha p^0)$ (Eq. (5) in \cite Bohn:2014xxa) as
 * \begin{equation}
 *   \frac{d \ln(\alpha p^0)}{d t} = -\Pi^i \partial_i \alpha
 *     + \alpha K_{ij} \Pi^i \Pi^j
 *   \text{.}
 * \end{equation}
 */
template <typename DataType, size_t Dim, typename Frame>
void geodesic_equation(
    // Output time derivs
    gsl::not_null<tnsr::I<DataType, Dim, Frame>*> dt_x,
    gsl::not_null<tnsr::i<DataType, Dim, Frame>*> dt_pi,
    gsl::not_null<Scalar<DataType>*> dt_lnp0,
    // Current state
    const tnsr::I<DataType, Dim, Frame>& x,
    const tnsr::i<DataType, Dim, Frame>& pi, const Scalar<DataType>& lnp0,
    // Background spacetime
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, Dim, Frame>& deriv_lapse,
    const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::iJ<DataType, Dim, Frame>& deriv_shift,
    const tnsr::II<DataType, Dim, Frame>& inv_spatial_metric,
    const tnsr::iJJ<DataType, Dim, Frame>& deriv_inv_spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature);

}  // namespace gr
