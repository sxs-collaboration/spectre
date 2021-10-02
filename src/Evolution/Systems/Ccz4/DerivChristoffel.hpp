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
 * \brief Computes the spatial derivative of the conformal spatial christoffel
 * symbols of the second kind
 *
 * \details Computes the derivative as:
 * \f{align}
 *     \partial_k \tilde{\Gamma}^m{}_{ij} &=
 *       -2 D_k{}^{ml} (D_{ijl} + D_{jil} - D_{lij}) +
 *       \tilde{\gamma}^{ml}(\partial_{(k} D_{i)jl} + \partial_{(k} D_{j)il} -
 *       \partial_{(k} D_{l)ij})
 * \f}
 * where \f$\tilde{\gamma}^{ij}\f$, \f$D_{ijk}\f$, \f$\partial_l D_{ijk}\f$, and
 * \f$D_k{}^{ij}\f$ are the inverse conformal spatial metric defined by
 * `Ccz4::Tags::InverseConformalMetric`, the CCZ4 auxiliary variable defined by
 * `Ccz4::Tags::FieldD`, its spatial derivative, and the CCZ4 identity defined
 * by `Ccz4::Tags::FieldDUp`.
 */
template <size_t Dim, typename Frame, typename DataType>
void deriv_conformal_christoffel_second_kind(
    const gsl::not_null<tnsr::iJkk<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::ijkk<DataType, Dim, Frame>& d_field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up);

template <size_t Dim, typename Frame, typename DataType>
tnsr::iJkk<DataType, Dim, Frame> deriv_conformal_christoffel_second_kind(
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::ijkk<DataType, Dim, Frame>& d_field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up);
/// @}
}  // namespace Ccz4
