// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr::surfaces {
/// @{
/*!
 * \ingroup SurfacesGroup
 * \brief Extrinsic curvature of a 2D `Strahlkorper` embedded in a 3D space.
 *
 * \details Implements Eq. (D.43) of Carroll's Spacetime and Geometry text.
 * Specifically,
 * \f$ K_{ij} = P^k_i P^l_j \nabla_{(k} S_{l)} \f$, where
 * \f$ P^k_i = \delta^k_i - S^k S_i \f$,
 * `grad_normal` is the quantity \f$ \nabla_k S_l \f$ returned by
 * `gr::surfaces::grad_unit_normal_one_form`, and `unit_normal_vector` is
 * \f$S^i = g^{ij} S_j\f$ where \f$S_j\f$ is the unit normal one form.
 * Not to be confused with the extrinsic curvature of a 3D spatial slice
 * embedded in 3+1 spacetime.
 * Because gr::surfaces::grad_unit_normal_one_form is symmetric, this
 * can be expanded into
 * \f$ K_{ij} = \nabla_{i}S_{j} - 2 S^k S_{(i}\nabla_{j)}S_k
 * + S_i S_j S^k S^l \nabla_{k} n_{l}\f$.
 */
template <typename Frame>
void extrinsic_curvature(
    gsl::not_null<tnsr::ii<DataVector, 3, Frame>*> result,
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector);

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> extrinsic_curvature(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector);
/// @}
}  // namespace gr::surfaces
