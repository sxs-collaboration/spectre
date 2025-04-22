// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
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
 * \note In second-order Ccz4, we impose symmetry of index k and l
 * in \f$ \partial_l D_{kij}=\frac{1}{2}\partial_l \partial_k
 * \tilde{\gamma}_{ij} \f$, because partial derivatives commute and to use
 * `second_partial_derivatives()`. \f$ D_{kij} \f$ is evolved in
 * the first-order system so no such symmetry is imposed.
 */
template <typename DataType, size_t Dim, typename Frame, typename TensorType>
void deriv_conformal_christoffel_second_kind(
    const gsl::not_null<tnsr::iJkk<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d, const TensorType& d_field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up);

template <typename DataType, size_t Dim, typename Frame, typename TensorType>
tnsr::iJkk<DataType, Dim, Frame> deriv_conformal_christoffel_second_kind(
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d, const TensorType& d_field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up);

/// @}

/// @{
/*!
 * \brief Computes the spatial derivative of the contraction of the conformal
 * spatial Christoffel symbols of the second kind
 *
 * \details Computes the derivative as:
 *
 * \f{align}
 *     \partial_k \tilde{\Gamma}^i &= -2 D_k{}^{jl} \tilde{\Gamma}^i_{jl} +
 *       \tilde{\gamma}^{jl} \partial_k \tilde{\Gamma}^i_{jl}
 * \f}
 *
 * where \f$\tilde{\gamma}^{ij}\f$ is the inverse conformal spatial metric
 * defined by `Ccz4::Tags::InverseConformalMetric`, \f$D_k{}^{ij}\f$ is the CCZ4
 * identity defined by `Ccz4::Tags::FieldDUp`, \f$\tilde{\Gamma}^k_{ij}\f$ is
 * the conformal spatial Christoffel symbols of the second kind defined by
 * `Ccz4::Tags::ConformalChristoffelSecondKind`, and
 * \f$\partial_k \tilde{\Gamma}^k_{ij}\f$ is its spatial derivative defined by
 * `Ccz4::Tags::DerivConformalChristoffelSecondKind`.
 */
template <typename DataType, size_t Dim, typename Frame>
void deriv_contracted_conformal_christoffel_second_kind(
    const gsl::not_null<tnsr::iJ<DataType, Dim, Frame>*> result,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind,
    const tnsr::iJkk<DataType, Dim, Frame>&
        d_conformal_christoffel_second_kind);

template <typename DataType, size_t Dim, typename Frame>
tnsr::iJ<DataType, Dim, Frame>
deriv_contracted_conformal_christoffel_second_kind(
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::Ijj<DataType, Dim, Frame>& conformal_christoffel_second_kind,
    const tnsr::iJkk<DataType, Dim, Frame>&
        d_conformal_christoffel_second_kind);
/// @}
}  // namespace Ccz4
