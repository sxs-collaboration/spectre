// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes Newman Penrose quantity \f$\Psi_4\f$ using the characteristic
 * field U\f$^{8+}\f$ and complex vector \f$\bar{m}^i\f$.
 *
 * \details Computes \f$\Psi_4\f$ as: \f$\Psi_4 =
 * U^{8+}_{ij}\bar{m}^i\bar{m}^j\f$ with the characteristic field
 * \f$U^{8+} = (P^{(a}_i P^{b)}_j - \frac{1}{2}P_{ij}P^{ab})
 * (E_{ab} - \epsilon_a^{cd}n_dB_{cb}\f$)
 * and \f$\bar{m}^i\f$ = \f$\frac{(x^i + iy^i)}{\sqrt{2}}\f$. \f$x^i\f$ and
 * \f$y^i\f$ are normalized unit vectors in the frame Frame.
 *
 */
template <typename Frame>
void psi_4(const gsl::not_null<Scalar<ComplexDataVector>*> psi_4_result,
           const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
           const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
           const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
           const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
           const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
           const tnsr::I<DataVector, 3, Frame>& inertial_coords);

template <typename Frame>
Scalar<ComplexDataVector> psi_4(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords);
/// @}

}  // namespace gr
