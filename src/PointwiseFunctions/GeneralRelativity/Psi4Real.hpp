// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4.hpp"
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
 * \brief Computes the real part of the Newman Penrose quantity \f$\Psi_4\f$
 * using  \f$\Psi_4[Real] = -0.5*U^{8+}_{ij}*(x^ix^j - y^iy^j)\f$.
 */
template <typename Frame>
void psi_4_real(
    const gsl::not_null<Scalar<DataVector>*> psi_4_real_result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords);

template <typename Frame>
Scalar<DataVector> psi_4_real(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords);

namespace Tags {
/// Computes the real part of the Newman Penrose quantity \f$\Psi_4\f$ using
/// \f$\Psi_4[Real] = -0.5*U^{8+}_{ij}*(x^ix^j - y^iy^j)\f$.
///
/// Can be retrieved using `gr::Tags::Psi4Real`
template <typename Frame>
struct Psi4RealCompute : Psi4Real<DataVector>, db::ComputeTag {
  using argument_tags = tmpl::list<
      gr::Tags::SpatialRicci<DataVector, 3, Frame>,
      gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame>,
      ::Tags::deriv<gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame>,
                    tmpl::size_t<3>, Frame>,
      gr::Tags::SpatialMetric<DataVector, 3, Frame>,
      gr::Tags::InverseSpatialMetric<DataVector, 3, Frame>,
      domain::Tags::Coordinates<3, Frame>>;

  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::ijj<DataVector, 3, Frame>&,
      const tnsr::ii<DataVector, 3, Frame>&,
      const tnsr::II<DataVector, 3, Frame>&,
      const tnsr::I<DataVector, 3, Frame>&)>(&psi_4_real<Frame>);
  using base = Psi4Real<DataVector>;
};
}  // namespace Tags
}  // namespace gr
