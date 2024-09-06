// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

/// @{
/*!
 * \brief Surface integrand for the ADM linear momentum calculation.
 *
 * We define the ADM linear momentum integral as (see Eqs. 19-20 in
 * \cite Ossokine2015yla):
 *
 * \begin{equation}
 *   P_\text{ADM}^i = \frac{1}{8\pi}
 *                    \oint_{S_\infty} \psi^10 \Big(
 *                      K^{ij} - K \gamma^{ij}
 *                    \Big) \, dS_j.
 * \end{equation}
 *
 * \note For consistency with `adm_linear_momentum_volume_integrand`, this
 * integrand needs to be contracted with the Euclidean face normal and
 * integrated with the Euclidean area element.
 *
 * \param result output pointer
 * \param conformal_factor the conformal factor $\psi$
 * \param inv_spatial_metric the inverse spatial metric $\gamma^{ij}$
 * \param inv_extrinsic_curvature the inverse extrinsic curvature $K^{ij}$
 * \param trace_extrinsic_curvature the trace of the extrinsic curvature $K$
 */
void adm_linear_momentum_surface_integrand(
    gsl::not_null<tnsr::II<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature);

/// Return-by-value overload
tnsr::II<DataVector, 3> adm_linear_momentum_surface_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature);
/// @}

/// @{
/*!
 * \brief Volume integrand for ADM linear momentum calculation defined as (see
 * Eq. 20 in \cite Ossokine2015yla):
 *
 * \begin{equation}
 *   P_\text{ADM}^i = - \frac{1}{8\pi}
 *                      \int_{V_\infty} \Big(
 *                        \bar\Gamma^i_{jk} P^{jk}
 *                        + \bar\Gamma^j_{jk} P^{jk}
 *                        - 2 \bar\gamma_{jk} P^{jk} \bar\gamma^{il}
 *                                                   \partial_l(\ln\psi)
 *                      \Big) \, dV,
 * \end{equation}
 *
 * where $1/(8\pi) P^{jk}$ is the result from
 * `adm_linear_momentum_surface_integrand`.
 *
 * \note For consistency with `adm_linear_momentum_surface_integrand`, this
 * integrand needs to be integrated with the Euclidean volume element.
 *
 * \param result output pointer
 * \param surface_integrand the quantity $1/(8\pi) P^{ij}$ (result of
 * `adm_linear_momentum_surface_integrand`)
 * \param conformal_factor the conformal factor $\psi$
 * \param deriv_conformal_factor the gradient of the conformal factor
 * $\partial_i\psi$
 * \param conformal_metric the conformal metric $\bar\gamma_{ij}$
 * \param inv_conformal_metric the inverse conformal metric $\bar\gamma^{ij}$
 * \param conformal_christoffel_second_kind the conformal christoffel symbol
 * $\bar\Gamma^i_{jk}$
 * \param conformal_christoffel_contracted the contracted conformal christoffel
 * symbol $\bar\Gamma_i$
 */
void adm_linear_momentum_volume_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::II<DataVector, 3>& surface_integrand,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted);

/// Return-by-value overload
tnsr::I<DataVector, 3> adm_linear_momentum_volume_integrand(
    const tnsr::II<DataVector, 3>& surface_integrand,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted);
/// @}

}  // namespace Xcts
