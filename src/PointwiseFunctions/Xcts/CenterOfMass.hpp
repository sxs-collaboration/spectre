// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

/// @{
/*!
 * \brief Surface integrand for the center of mass calculation.
 *
 * We define the center of mass integral as (see Eq. 25 in
 * \cite Ossokine2015yla):
 *
 * \begin{equation}
 *   C_\text{CoM}^i = \frac{3}{8 \pi M_\text{ADM}}
 *               \oint_{S_\infty} \psi^4 n^i \, dA,
 * \end{equation}
 *
 * where $n^i = x^i / r$ and $r = \sqrt{x^2 + y^2 + z^2}$.
 *
 * \note We don't include the ADM mass $M_{ADM}$ in this integrand. After
 * integrating the result of this function, you have to divide by $M_{ADM}$.
 *
 * \note For consistency with `center_of_mass_volume_integrand`, this
 * integrand needs to be integrated with the Euclidean area element.
 *
 * \see `Xcts::adm_mass_surface_integrand`
 *
 * \warning This integral assumes that the conformal metric falls off to
 * flatness faster than $1/r^2$. That means that it cannot be directly used
 * with the Kerr-Schild metric, which falls off as $1/r$. This is not a problem
 * for XCTS with Superposed Kerr-Schild (SKS) because of the exponential
 * fall-off terms.
 *
 * \param result output pointer
 * \param conformal_factor the conformal factor $\psi$
 * \param coords the inertial coordinates $x^i$
 */
void center_of_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& coords);

/// Return-by-value overload
tnsr::I<DataVector, 3> center_of_mass_surface_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& coords);
/// @}

/// @{
/*!
 * \brief Volume integrand for the center of mass calculation.
 *
 * We cast the center of mass as an infinite volume integral by applying Gauss'
 * law on the surface integral defined in `center_of_mass_surface_integrand`:
 *
 * \begin{equation}
 *   C_\text{CoM}^i = \frac{3}{8 \pi M_\text{ADM}}
 *                    \int_{V_\infty} \Big(
 *                      4 \psi^3 \partial_j \psi n^i n^j
 *                      + \frac{2}{r} \psi^4 n^i
 *                    \Big) dV
 *                  = \frac{3}{4 \pi M_\text{ADM}}
 *                    \int_{V_\infty} \frac{1}{r^2} \Big(
 *                      2 \psi^3 \partial_j \psi x^i x^j
 *                      + \psi^4 x^i
 *                    \Big) dV,
 * \end{equation}
 *
 * where $n^i = x^i / r$ and $r = \sqrt{x^2 + y^2 + z^2}$.
 *
 * \note For consistency with `center_of_mass_surface_integrand`, this
 * integrand needs to be integrated with the Euclidean volume element.
 *
 * \see `center_of_mass_surface_integrand`
 *
 * \param result output pointer
 * \param conformal_factor the conformal factor $\psi$
 * \param deriv_conformal_factor the gradient of the conformal factor
 * $\partial_i \psi$
 * \param coords the inertial coordinates $x^i$
 */
void center_of_mass_volume_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& deriv_conformal_factor,
    const tnsr::I<DataVector, 3>& coords);

/// Return-by-value overload
tnsr::I<DataVector, 3> center_of_mass_volume_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& deriv_conformal_factor,
    const tnsr::I<DataVector, 3>& coords);
/// @}

}  // namespace Xcts
