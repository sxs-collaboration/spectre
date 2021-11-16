// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \brief Computes the sum of the Ricci scalar and twice the divergence of the
 * upper spatial Z4 constraint
 *
 * \details Computes the expression as:
 *
 * \f{align}
 *     R + 2 \nabla_k Z^k &=
 *         \phi^2 \tilde{\gamma}^{ij} (R_{ij} + \nabla_i Z_j + \nabla_j Z_i)
 * \f}
 *
 * where \f$R\f$ is the spatial Ricci scalar, \f$Z^i\f$ is the upper spatial Z4
 * constraint defined by `Ccz4::Tags::Z4ConstraintUp`, \f$phi^2\f$ is the square
 * of the conformal factor defined by `Ccz4::Tags::ConformalFactorSquared`,
 * \f$\tilde{\gamma}^{ij}\f$ is the inverse conformal spatial metric defined by
 * `Ccz4::Tags::InverseConformalMetric`, \f$R_{ij}\f$ is the spatial Ricci
 * tensor defined by `Ccz4::Tags::Ricci`, and \f$\nabla_j Z_i\f$ is the gradient
 * of the spatial Z4 constraint defined by `Ccz4::Tags::GradZ4Constraint`.
 */
template <size_t Dim, typename Frame, typename DataType>
void ricci_scalar_plus_divergence_z4_constraint(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& conformal_factor_squared,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& spatial_ricci_tensor,
    const tnsr::ij<DataType, Dim, Frame>& grad_spatial_z4_constraint);

template <size_t Dim, typename Frame, typename DataType>
Scalar<DataType> ricci_scalar_plus_divergence_z4_constraint(
    const Scalar<DataType>& conformal_factor_squared,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& spatial_ricci_tensor,
    const tnsr::ij<DataType, Dim, Frame>& grad_spatial_z4_constraint);
/// @}
}  // namespace Ccz4
