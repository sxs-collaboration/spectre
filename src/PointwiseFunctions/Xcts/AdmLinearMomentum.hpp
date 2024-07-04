// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

#include <cmath>

#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

namespace Xcts {

/// @{
/*!
 * \brief Surface integrand for ADM linear momentum calculation defined as (see
 * Eq. 20 in \cite Ossokine2015yla):
 *
 * \begin{equation}
 *   \frac{1}{8\pi} \psi^{10} (K^{ij} - K \gamma^{ij})
 * \end{equation}
 *
 * \param result output buffer for the surface integrand
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
 *   - \frac{1}{8\pi} (
 *       \bar\Gamma^i_{jk} P^{jk}
 *       + \bar\Gamma^j_{jk} P^{jk}
 *       - 2 \bar\gamma_{jk} P^{jk} \bar\gamma^{il} \partial_l(\ln\psi)
 *   ),
 * \end{equation}
 *
 * where $\frac{1}{8\pi} P^{jk}$ is the result from
 * `adm_linear_momentum_surface_integrand`.
 *
 * Note that we are including the negative sign in the integrand.
 *
 * \param result output buffer for the surface integrand
 * \param surface_integrand the result of
 * `adm_linear_momentum_surface_integrand`
 * \param conformal_factor the conformal factor $\psi$
 * \param conformal_factor_deriv the derivative of the conformal factor
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
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted);

/// Return-by-value overload
tnsr::I<DataVector, 3> adm_linear_momentum_volume_integrand(
    const tnsr::II<DataVector, 3>& surface_integrand,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted);
/// @}

}  // namespace Xcts
