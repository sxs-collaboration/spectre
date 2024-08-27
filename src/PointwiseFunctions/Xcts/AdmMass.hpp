// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

/// @{
/*!
 * \brief Surface integrand for the ADM mass calculation.
 *
 * We define the ADM mass integral as (see Eq. 3.139 in \cite BaumgarteShapiro):
 *
 * \begin{equation}
 *   M_{ADM}
 *   = \int_{S_\infty} \frac{1}{16\pi} (
 *     \bar\gamma^{jk} \bar\Gamma^i_{jk}
 *     - \bar\gamma^{ij} \bar\Gamma_{j}
 *     - 8 \bar\gamma^{ij} \partial_j \psi
 *   ) d\bar{S}_i.
 * \end{equation}
 *
 * Note that we don't use the other versions presented in \cite BaumgarteShapiro
 * of this integral because they make assumptions like $\bar\gamma = 1$,
 * $\bar\Gamma^i_{ij} = 0$ and fast fall-off of the conformal metric.
 *
 * \param result output buffer for the surface integrand
 * \param deriv_conformal_factor the partial derivatives of the conformal factor
 * $\partial_i \psi$
 * \param inv_conformal_metric the inverse conformal metric $\bar\gamma^{ij}$
 * \param conformal_christoffel_second_kind the conformal christoffel symbol
 * $\bar\Gamma^i_{jk}$
 * \param conformal_christoffel_contracted the conformal christoffel symbol
 * contracted in its first two indices $\bar\Gamma_{i} = \bar\Gamma^j_{ij}$
 */
void adm_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted);

/// Return-by-value overload
tnsr::I<DataVector, 3> adm_mass_surface_integrand(
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted);
/// @}

}  // namespace Xcts
