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
 *   M_\text{ADM} = \frac{1}{16\pi}
 *                  \oint_{S_\infty} \Big(
 *                     \bar\gamma^{jk} \bar\Gamma^i_{jk}
 *                     - \bar\gamma^{ij} \bar\Gamma_{j}
 *                     - 8 \bar\gamma^{ij} \partial_j \psi
 *                  \Big) d\bar{S}_i.
 * \end{equation}
 *
 * \note We don't use the other versions presented in \cite BaumgarteShapiro of
 * this integral because they make assumptions like $\bar\gamma = 1$,
 * $\bar\Gamma^i_{ij} = 0$ and faster fall-off of the conformal metric.
 *
 * \note For consistency with `adm_mass_volume_integrand`, this integrand needs
 * to be contracted with the conformal face normal and integrated with the
 * conformal area element.
 *
 * \param result output pointer
 * \param deriv_conformal_factor the gradient of the conformal factor
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

/// @{
/*!
 * \brief Volume integrand for the ADM mass calculation.
 *
 * We cast the ADM mass as an infinite volume integral by applying Gauss' law on
 * the surface integral defined in `adm_mass_surface_integrand`:
 *
 * \begin{equation}
 *   M_\text{ADM} = \frac{1}{16\pi}
 *                  \int_{V_\infty} \Big(
 *                    \partial_i \bar\gamma^{jk} \bar\Gamma^i_{jk}
 *                    + \bar\gamma^{jk} \partial_i \bar\Gamma^i_{jk}
 *                    + \bar\Gamma_l \bar\gamma^{jk} \bar\Gamma^l_{jk}
 *                    - \partial_i \bar\gamma^{ij} \bar\Gamma_j
 *                    - \bar\gamma^{ij} \partial_i \bar\Gamma_j
 *                    - \bar\Gamma_l \bar\gamma^{lj} \bar\Gamma_j
 *                    - 8 \bar D^2 \psi
 *                  \Big) d\bar{V},
 * \end{equation}
 *
 * where we can use the Hamiltonian constraint (Eq. 3.37 in
 * \cite BaumgarteShapiro) to replace $8 \bar D^2 \psi$ with
 *
 * \begin{equation}
 *   8 \bar D^2 \psi = \psi \bar R + \frac{2}{3} \psi^5 K^2
 *                     - \frac{1}{4} \psi^5 \frac{1}{\alpha^2}
 *                         \Big[ (\bar L \beta)_{ij} - \bar u_{ij} \Big]
 *                         \Big[ (\bar L \beta)^{ij} - \bar u^{ij} \Big]
 *                     - 16\pi \psi^5 \rho.
 * \end{equation}
 *
 * \note This is similar to Eq. 3.149 in \cite BaumgarteShapiro, except that
 * here we don't assume $\bar\gamma = 1$.
 *
 * \note For consistency with `adm_mass_surface_integrand`, this integrand needs
 * to be integrated with the conformal volume element.
 *
 * \param result output pointer
 * \param conformal_factor the conformal factor
 * \param conformal_ricci_scalar the conformal Ricci scalar $\bar R$
 * \param trace_extrinsic_curvature the extrinsic curvature trace $K$
 * \param longitudinal_shift_minus_dt_conformal_metric_over_lapse_square the
 * quantity computed in
 * `Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare`
 * \param energy_density the energy density $\rho$
 * \param inv_conformal_metric the inverse conformal metric $\bar\gamma^{ij}$
 * \param deriv_inv_conformal_metric the gradient of the inverse conformal
 * metric $\partial_i \bar\gamma^{jk}$
 * \param conformal_christoffel_second_kind the conformal christoffel symbol
 * $\bar\Gamma^i_{jk}$
 * \param conformal_christoffel_contracted the conformal christoffel symbol
 * contracted in its first two indices $\bar\Gamma_{i} = \bar\Gamma^j_{ij}$
 * \param deriv_conformal_christoffel_second_kind the gradient of the conformal
 * christoffel symbol $\partial_i \bar\Gamma^j_{kl}$
 */
void adm_mass_volume_integrand(
    gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& energy_density,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::iJK<DataVector, 3>& deriv_inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::iJkk<DataVector, 3>& deriv_conformal_christoffel_second_kind);

/// Return-by-value overload
Scalar<DataVector> adm_mass_volume_integrand(
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& energy_density,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::iJK<DataVector, 3>& deriv_inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::iJkk<DataVector, 3>& deriv_conformal_christoffel_second_kind);
/// @}

}  // namespace Xcts
