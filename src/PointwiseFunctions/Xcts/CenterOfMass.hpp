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
 *   C_{CoM}^i
 *   = \frac{1}{M_{ADM}} \int_{S_\infty} \frac{3}{8\pi} \psi^4 n^i \, d\bar{A}.
 * \end{equation}
 *
 * Note that we don't include the ADM mass $M_{ADM}$ in this integrand. After
 * integrating the result of this function, you have to divide by $M_{ADM}$.
 * See `Xcts::adm_mass_surface_integrand` for details on how to calculate
 * $M_{ADM}$.
 *
 * \warning This integral assumes that the conformal metric falls off to
 * flatness faster than $1/r^2$. That means that it cannot be directly used
 * with the Kerr-Schild metric, which falls off as $1/r$. This is not a problem
 * for XCTS with Superposed Kerr-Schild (SKS) because of the exponential
 * fall-off terms.
 *
 * \param result output buffer for the surface integrand
 * \param conformal_factor the conformal factor $\psi$
 * \param unit_normal the outward-pointing unit normal $n^i = x^i / r$
 */
void center_of_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& unit_normal);

/// Return-by-value overload
tnsr::I<DataVector, 3> center_of_mass_surface_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& unit_normal);
/// @}

}  // namespace Xcts
