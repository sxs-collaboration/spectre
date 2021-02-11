// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {
/*!
 * \brief The longitudinal operator, or vector gradient, \f$(L\beta)^{ij}\f$
 *
 * Computes the longitudinal operator
 *
 * \f{equation}
 * (L\beta)^{ij} = \nabla^i \beta^j + \nabla^j \beta^i -
 * \frac{2}{3}\gamma^{ij}\nabla_k\beta^k
 * \f}
 *
 * of a vector field \f$\beta^i\f$, where \f$\nabla\f$ denotes the covariant
 * derivative w.r.t. the metric \f$\gamma\f$ (see e.g. Eq. (3.50) in
 * \cite BaumgarteShapiro). Note that in the XCTS equations the longitudinal
 * operator is typically applied to conformal quantities and w.r.t. the
 * conformal metric \f$\bar{\gamma}\f$.
 *
 * In terms of the symmetric "strain" quantity
 * \f$B_{ij}=\nabla_{(i}\gamma_{j)k}\beta^k\f$ the longitudinal operator is:
 *
 * \f{equation}
 * (L\beta)^{ij} = 2\left(\gamma^{ik}\gamma^{jl} -
 * \frac{1}{3} \gamma^{jk}\gamma^{kl}\right) B_{kl}
 * \f}
 */
template <typename DataType>
void longitudinal_operator(gsl::not_null<tnsr::II<DataType, 3>*> result,
                           const tnsr::ii<DataType, 3>& strain,
                           const tnsr::II<DataType, 3>& inv_metric) noexcept;

/*!
 * \brief The conformal longitudinal operator \f$(L\beta)^{ij}\f$ on a flat
 * conformal metric in Cartesian coordinates \f$\gamma_{ij}=\delta_{ij}\f$
 *
 * \see `Xcts::longitudinal_operator`
 */
template <typename DataType>
void longitudinal_operator_flat_cartesian(
    gsl::not_null<tnsr::II<DataType, 3>*> result,
    const tnsr::ii<DataType, 3>& strain) noexcept;
}  // namespace Xcts
