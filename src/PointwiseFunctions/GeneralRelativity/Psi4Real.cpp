// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace gr {

template <typename Frame>
void psi_4_real(
    const gsl::not_null<Scalar<DataVector>*> psi_4_real_result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords) {
  get(*psi_4_real_result) = real(get(
      psi_4(spatial_ricci, extrinsic_curvature, cov_deriv_extrinsic_curvature,
            spatial_metric, inverse_spatial_metric, inertial_coords)));
}

template <typename Frame>
Scalar<DataVector> psi_4_real(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::ijj<DataVector, 3, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame>& inverse_spatial_metric,
    const tnsr::I<DataVector, 3, Frame>& inertial_coords) {
  auto psi_4_real_result = make_with_value<Scalar<DataVector>>(
      get<0, 0>(inverse_spatial_metric),
      std::numeric_limits<double>::signaling_NaN());
  psi_4_real(make_not_null(&psi_4_real_result), spatial_ricci,
             extrinsic_curvature, cov_deriv_extrinsic_curvature, spatial_metric,
             inverse_spatial_metric, inertial_coords);
  return psi_4_real_result;
}
}  // namespace gr

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template Scalar<DataVector> gr::psi_4_real(                             \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_ricci,          \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature,    \
      const tnsr::ijj<DataVector, 3, FRAME(data)>&                        \
          cov_deriv_extrinsic_curvature,                                  \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,         \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_spatial_metric, \
      const tnsr::I<DataVector, 3, FRAME(data)>& inertial_coords);        \
  template void gr::psi_4_real(                                           \
      const gsl::not_null<Scalar<DataVector>*> psi_4_real_result,         \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_ricci,          \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature,    \
      const tnsr::ijj<DataVector, 3, FRAME(data)>&                        \
          cov_deriv_extrinsic_curvature,                                  \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,         \
      const tnsr::II<DataVector, 3, FRAME(data)>& inverse_spatial_metric, \
      const tnsr::I<DataVector, 3, FRAME(data)>& inertial_coords);

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial))

#undef FRAME
#undef INSTANTIATE
