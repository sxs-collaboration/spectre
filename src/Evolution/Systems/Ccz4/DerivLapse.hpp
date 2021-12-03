// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \brief Computes the gradient of the gradient of the lapse.
 *
 * \details Computes the gradient of the gradient as:
 * \f{align}
 *     \nabla_i \nabla_j \alpha &= \alpha A_i A_j -
 *                 \alpha \Gamma^k{}_{ij} A_k + \alpha \partial_{(i} A_{j)}
 * \f}
 * where \f$\alpha\f$, \f$\Gamma^k{}_{ij}\f$, \f$A_i\f$, and
 * \f$\partial_j A_i\f$ are the lapse, spatial christoffel symbols of the second
 * kind, the CCZ4 auxiliary variable defined by `Ccz4::Tags::FieldA`, and its
 * spatial derivative, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
void grad_grad_lapse(
    const gsl::not_null<tnsr::ij<DataType, Dim, Frame>*> result,
    const Scalar<DataType>& lapse,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& field_a,
    const tnsr::ij<DataType, Dim, Frame>& d_field_a);

template <size_t Dim, typename Frame, typename DataType>
tnsr::ij<DataType, Dim, Frame> grad_grad_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& field_a,
    const tnsr::ij<DataType, Dim, Frame>& d_field_a);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the divergence of the lapse.
 *
 * \details Computes the divergence as:
 * \f{align}
 *     \nabla^i \nabla_i \alpha &= \phi^2 \tilde{\gamma}^{ij}
 *         (\nabla_i \nabla_j \alpha)
 * \f}
 * where \f$\phi\f$, \f$\tilde{\gamma}^{ij}\f$, and
 * \f$\nabla_i \nabla_j \alpha\f$ are the conformal factor, inverse conformal
 * spatial metric, and the gradient of the gradient of the lapse defined by
 * `Ccz4::Tags::ConformalFactor`, `Ccz4::Tags::InverseConformalMetric`, and
 * `Ccz4::Tags::GradGradLapse`, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
void divergence_lapse(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& conformal_factor_squared,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_metric,
    const tnsr::ij<DataType, Dim, Frame>& grad_grad_lapse);

template <size_t Dim, typename Frame, typename DataType>
Scalar<DataType> divergence_lapse(
    const Scalar<DataType>& conformal_factor_squared,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_metric,
    const tnsr::ij<DataType, Dim, Frame>& grad_grad_lapse);
/// @}
}  // namespace Ccz4
