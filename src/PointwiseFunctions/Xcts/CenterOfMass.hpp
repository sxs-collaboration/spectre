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
 * We define the center of mass integral as
 *
 * \begin{equation}
 *   C_\text{CoM}^i = \frac{3}{8 \pi M_\text{ADM}} \oint_{S_\infty} \Big[
 *                      (\psi - 1)^4 + 4 (\psi - 1)^3
 *                      + 6 (\psi - 1)^2 + 4 (\psi - 1)
 *                    \Big] n^i \, dA,
 * \end{equation}
 *
 * where $n^i = x^i / r$ and $r = \sqrt{x^2 + y^2 + z^2}$. We use $\psi-1$
 * instead of $\psi$ because it is the variable solved for in the XCTS system.
 * Expanding the integrand, we see that this is identical to
 *
 * \begin{equation}
 *   C_\text{CoM}^i = \frac{3}{8 \pi M_\text{ADM}}
 *               \oint_{S_\infty} (\psi^4 - 1) n^i \, dA,
 * \end{equation}
 *
 * Analytically, this is equivalent to the definition in Eq. (25) of
 * \cite Ossokine2015yla because
 * \begin{equation}
 *   \oint_{S_\infty} n^i \, dA = 0.
 * \end{equation}
 * Numerically, we have found that subtracting $1$ from $\psi^4$ results in less
 * round-off errors, leading to a more accurate center of mass.
 *
 * One way to interpret this integral is that we are summing over the unit
 * vectors $n^i$, rescaled by $\psi^4$, in all directions. If $\psi(\vec r)$ is
 * constant, no rescaling happens and all the unit vectors cancel out. If
 * $\psi(\vec r)$ is not constant, then $\vec C_\text{CoM}$ will emerge from the
 * difference of large numbers (sum of rescaled $n^i$ in each subdomain). With
 * larger and larger numbers being involved in this cancellation (i.e. with
 * increasing radius of $S_\infty$), we loose numerical accuracy. In other
 * words, we are seeking the subdominant terms. Since $\psi^4 \to 1$ in
 * conformal flatness, subtracting $1$ from it in the integrand makes the
 * numbers involved in this cancellation smaller, reducing this issue.
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
 * \param conformal_factor_minus_one the conformal factor $\psi - 1$
 * \param coords the inertial coordinates $x^i$
 */
void center_of_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& coords);

/// Return-by-value overload
tnsr::I<DataVector, 3> center_of_mass_surface_integrand(
    const Scalar<DataVector>& conformal_factor_minus_one,
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
 * \warning This integral currently suffers from significant round-off errors.
 * It is recommended to use only the surface integral if possible.
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
