// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
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
    const tnsr::ij<DataType, Dim, Frame>& d_field_a) noexcept;

template <size_t Dim, typename Frame, typename DataType>
tnsr::ij<DataType, Dim, Frame> grad_grad_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::i<DataType, Dim, Frame>& field_a,
    const tnsr::ij<DataType, Dim, Frame>& d_field_a) noexcept;
/// @}
}  // namespace Ccz4
