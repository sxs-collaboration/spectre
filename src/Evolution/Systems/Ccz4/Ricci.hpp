// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
/// @{
/*!
 * \brief Computes the spatial Ricci tensor
 *
 * \details Computes the spatial Ricci tensor as:
 *
 * \f{align}
 *     R_{ij} &=
 *       \partial_m \Gamma^m_{ij} - \partial_j \Gamma^m_{im} +
 *       \Gamma^l_{ij} \Gamma^m_{lm} - \Gamma^l_{im} \Gamma^m_{lj}
 * \f}
 *
 * where
 *
 * \f{align}
 *     \partial_k \Gamma^m_{ij} &=
 *       \partial_k \tilde{\Gamma}^m_{ij} +
 *       2 D_k{}^{ml} (\tilde{\gamma}_{jl} P_i + \tilde{\gamma}_{il} P_j -
 *                     \tilde{\gamma}_{ij} P_l)\nonumber\\
 *       & - 2 \tilde{\gamma}^{ml} (D_{kjl} P_i + D_{kil} P_j - D_{kij} P_l) -
 *       \tilde{\gamma}^{ml} (
 *         \tilde{\gamma}_{jl} \partial_{(k} P_{i)} +
 *         \tilde{\gamma}_{il} \partial_{(k} P_{j)} -
 *         \tilde{\gamma}_{ij} \partial_{(k} P_{l)})
 * \f}
 *
 * \f$\Gamma^k_{ij}\f$ is the spatial christoffel symbols of the second kind
 * defined by `Ccz4::Tags::ChristoffelSecondKind`,
 * \f$\partial_m \tilde{\Gamma}^k_{ij}\f$ is the spatial derivative of the
 * conformal spatial christoffel symbols of the second kind defined by
 * `Ccz4::Tags::DerivConformalChristoffelSecondKind`, \f$\tilde{\gamma}_{ij}\f$
 * is the conformal spatial metric defined by `Ccz4::Tags::ConformalMetric`,
 * \f$\tilde{\gamma}^{ij}\f$ is the inverse conformal spatial metric defined by
 * `Ccz4::Tags::InverseConformalMetric`, \f$D_{ijk}\f$ is the CCZ4 auxiliary
 * variable defined by `Ccz4::Tags::FieldD`, \f$D_k{}^{ij}\f$ is the CCZ4
 * identity defined by `Ccz4::Tags::FieldDUp`, \f$P_i\f$ is the CCZ4 auxiliary
 * variable defined by `Ccz4::Tags::FieldP`, and \f$\partial_j P_{i}\f$ is its
 * spatial derivative.
 *
 * After substituting in the full expressions for \f$\partial_m \Gamma^m_{ij}\f$
 * and \f$\partial_j \Gamma^m_{im}\f$ and commuting terms with common
 * coefficients, the full equation becomes and is implemented as:
 *
 *  \f{align}{
 *     R_{ij} &=
 *       \partial_m \tilde{\Gamma}^m_{ij} -
 *       \partial_j \tilde{\Gamma}^m_{im}\nonumber\\
 *       & + 2 D_m{}^{ml} (\tilde{\gamma}_{jl} P_i + \tilde{\gamma}_{il} P_j -
 *                         \tilde{\gamma}_{ij} P_l) -
 *       2 \tilde{\gamma}^{ml} (
 *         D_{mjl} P_i + D_{mil} P_j - D_{mij} P_l)\nonumber\\
 *       & - 2 D_j{}^{ml} (\tilde{\gamma}_{ml} P_i + \tilde{\gamma}_{il} P_m -
 *                         \tilde{\gamma}_{im} P_l) +
 *       2 \tilde{\gamma}^{ml} (
 *         D_{jml} P_i + D_{jil} P_m - D_{jim} P_l)\nonumber\\
 *       & - \tilde{\gamma}^{ml} (
 *         \tilde{\gamma}_{jl} \partial_{(m} P_{i)} +
 *         \tilde{\gamma}_{il} \partial_{(m} P_{j)} -
 *         \tilde{\gamma}_{ij} \partial_{(m} P_{l)}) +
 *       \tilde{\gamma}^{ml} (
 *         \tilde{\gamma}_{ml} \partial_{(j} P_{i)} +
 *         \tilde{\gamma}_{il} \partial_{(j} P_{m)} -
 *         \tilde{\gamma}_{im} \partial_{(j} P_{l)})\nonumber\\
 *       & + \Gamma^l_{ij} \Gamma^m_{lm} - \Gamma^l_{im} \Gamma^m_{lj}
 * \f}
 *
 */
template <size_t Dim, typename Frame, typename DataType>
void spatial_ricci_tensor(
    const gsl::not_null<tnsr::ii<DataType, Dim, Frame>*> result,
    const gsl::not_null<tnsr::ij<DataType, Dim, Frame>*> buffer,
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::iJkk<DataType, Dim, Frame>& d_conformal_christoffel_second_kind,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::ij<DataType, Dim, Frame>& d_field_p);

template <size_t Dim, typename Frame, typename DataType>
tnsr::ii<DataType, Dim, Frame> spatial_ricci_tensor(
    const tnsr::Ijj<DataType, Dim, Frame>& christoffel_second_kind,
    const tnsr::iJkk<DataType, Dim, Frame>& d_conformal_christoffel_second_kind,
    const tnsr::ii<DataType, Dim, Frame>& conformal_spatial_metric,
    const tnsr::II<DataType, Dim, Frame>& inverse_conformal_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& field_d,
    const tnsr::iJJ<DataType, Dim, Frame>& field_d_up,
    const tnsr::i<DataType, Dim, Frame>& field_p,
    const tnsr::ij<DataType, Dim, Frame>& d_field_p);
/// @}
}  // namespace Ccz4
